import argparse
import hashlib
import itertools
import math
import os, inspect
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

try:
    from lora_diffusion import (
        extract_lora_ups_down,
        inject_trainable_lora,
        safetensors_available,
        save_lora_weight,
        save_safeloras,
        weight_apply_lora,
    )
except ImportError:
    raise ImportError(
        "To use LORA, please install the lora-diffusion library: `pip install git+https://github.com/cloneofsimo/lora.git`."
    )


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        use_filename_as_label=False,
        use_txt_as_label=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(self.instance_data_root.glob("*.jpg")) + list(
            self.instance_data_root.glob("*.png")
        )
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self.use_filename_as_label = use_filename_as_label
        self.use_txt_as_label = use_txt_as_label
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.glob("*.jpg")) + list(
                self.class_data_root.glob("*.png")
            )
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.ColorJitter(0.2, 0.1),
                # transforms.RandomHorizontalFlip(1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        path = self.instance_images_path[index % self.num_instance_images]
        prompt = (
            get_filename(path) if self.use_filename_as_label else self.instance_prompt
        )
        prompt = get_label_from_txt(path) if self.use_txt_as_label else prompt

        # print("\n# prompt", prompt)

        instance_image = Image.open(path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--use_filename_as_label",
        action="store_true",
        help="Uses the filename as the image labels instead of the instance_prompt, useful for regularization when training for styles with wide image variance",
    )
    parser.add_argument(
        "--use_txt_as_label",
        action="store_true",
        help="Uses the filename.txt file's content as the image labels instead of the instance_prompt, useful for regularization when training for styles with wide image variance",
    )
    parser.add_argument(
        "--negative_prompt_for_train",
        type=str,
        default="",
        help="The negative prompt to use for training the model.",
    )
    parser.add_argument(
        "--train_unet", action="store_true", help="Whether to train the unet"
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_vae", action="store_true", help="Whether to train the vae"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="whether to use LoRA for gradient checkpointing.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA approximation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_text",
        type=float,
        default=1e-6,
        help="Initial learning rate for text encoder (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_vae",
        type=float,
        default=1e-6,
        help="Initial learning rate for vae (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--use_velo",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--log_with", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument("--save_model_every_n_epochs", type=int, default=1000)
    parser.add_argument(
        "--resume_unet",
        type=str,
        default=None,
        help=("File path for unet lora to resume training."),
    )
    parser.add_argument(
        "--resume_text_encoder",
        type=str,
        default=None,
        help=("File path for text encoder lora to resume training."),
    )
    parser.add_argument(
        "--use_xformers", action="store_true", help="Whether or not to use xformers"
    )
    parser.add_argument(
        "--auto_test_model",
        action="store_true",
        help="Whether or not to automatically test the model after saving it",
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default="A photo of a cat",
        help="The prompt to use for testing the model.",
    )
    parser.add_argument(
        "--test_prompts_file",
        type=str,
        default=None,
        help="The file containing the prompts to use for testing the model.example: test_prompts.txt, each line is a prompt",
    )
    parser.add_argument(
        "--negative_prompt_for_test",
        type=str,
        default="",
        help="The negative prompt to use for testing the model.",
    )
    parser.add_argument(
        "--test_width",
        type=int,
        default=512,
        help="The width to use for testing the model.",
    )
    parser.add_argument(
        "--test_height",
        type=int,
        default=512,
        help="The height to use for testing the model.",
    )
    parser.add_argument(
        "--test_seed",
        type=int,
        default=42,
        help="The seed to use for testing the model.",
    )
    parser.add_argument(
        "--test_num_per_prompt",
        type=int,
        default=1,
        help="The number of images to generate per prompt.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.instance_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")

    return args


# turns a path into a filename without the extension
def get_filename(path):
    return path.stem


def get_label_from_txt(path):
    txt_path = path.with_suffix(".txt")  # get the path to the .txt file
    if txt_path.exists():
        with open(txt_path, "r") as f:
            return f.read()
    else:
        return ""


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def test_model(folder, pipeline, args):
    if args.test_prompts_file is not None:
        with open(args.test_prompts_file, "r") as f:
            prompts = f.read().splitlines()
    else:
        prompts = [args.test_prompt]

    test_path = os.path.join(folder, "test")
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    print("Testing the model...")
    from diffusers import DDIMScheduler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    def dummy(images, **kwargs):
        return images, False
    
    pipeline.safety_checker = dummy
    pipeline.enable_attention_slicing()
    pipeline.set_use_memory_efficient_attention_xformers(True)
    pipeline.set_progress_bar_config(disable=True)
    
    pipeline = pipeline.to(device)

    torch.manual_seed(args.test_seed)
    with torch.autocast("cuda"):
        for prompt in prompts:
            print(f"Generating test images for prompt: {prompt}")
            test_images = pipeline(
                prompt=prompt,
                width=args.test_width,
                height=args.test_height,
                negative_prompt=args.negative_prompt_for_test,
                num_inference_steps=30,
                num_images_per_prompt=args.test_num_per_prompt,
            ).images

            for index, image in enumerate(test_images):
                image.save(f"{test_path}/{prompt}_{index}.png")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Test completed.The examples are saved in {test_path}")
    del pipeline


def merge_and_remove_lora(model, ancestor_class, alpha=1.0):
    # merge lora parameters to model
    for _fmodule in model.modules():
        if _fmodule.__class__.__name__ not in ancestor_class:
            continue

        for name, _module in _fmodule.named_modules():
            if _module.__class__.__name__ != "LoraInjectedLinear":
                continue

            # create new linear layer
            _linear = torch.nn.Linear(
                _module.linear.in_features,
                _module.linear.out_features,
                bias=_module.linear.bias is not None,
            )
            # copy linear weight
            linear_weight = _module.linear.weight

            # copy lora weight
            up_weight = _module.lora_up.weight
            down_weight = _module.lora_down.weight
            lora_weight: torch.Tensor = up_weight @ down_weight

            # merge lora weight to linear weight
            _linear.weight = torch.nn.Parameter(linear_weight + lora_weight * alpha)

            print("Merging Lora to Linear...")
            print(_module)
            # replace lora injected layer with new linear layer
            _fmodule[name] = _linear
            print(_module)


def save_model(accelerator, unet, text_encoder, args, epoch, step):
    folder = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if not os.path.exists(folder):
        os.makedirs(folder)

    print("Saving Model Checkpoint...")
    print("Directory: " + folder)

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
            inspect.signature(accelerator.unwrap_model).parameters.keys()
        )
        extra_args = {"keep_fp32_wrapper": True} if accepts_keep_fp32_wrapper else {}
        unet = accelerator.unwrap_model(unet, **extra_args)
        text_encoder = accelerator.unwrap_model(text_encoder, **extra_args)
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            text_encoder=text_encoder,
            revision=args.revision,
        )

        if args.use_lora:
            lora_path = os.path.join(folder, "lora_weights")
            if not os.path.exists(lora_path):
                os.makedirs(lora_path)

            print(f"save lora weights")
            loras = {}

            # save unet lora parameters
            filename_lora_unet = f"{lora_path}/lora_e{epoch}_s{step}.pt"
            save_lora_weight(pipeline.unet, filename_lora_unet)
            loras["unet"] = (pipeline.unet, {"CrossAttention", "Attention", "GEGLU"})

            # save text encoder lora parameters
            if args.train_text_encoder:
                filename_lora_clip = (
                    f"{lora_path}/lora_e{epoch}_s{step}.text_encoder.pt"
                )
                save_lora_weight(
                    pipeline.text_encoder,
                    filename_lora_clip,
                    target_replace_module=["CLIPAttention"],
                )
                loras["text_encoder"] = (pipeline.text_encoder, {"CLIPAttention"})

            # save lora parameters
            filename_lora = f"{lora_path}/lora_e{epoch}_s{step}.safetensors"
            save_safeloras(loras, filename_lora)

        else:
            # save full model
            pipeline.save_pretrained(folder)

        if args.auto_test_model:
            test_model(folder, pipeline, args)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.push_to_hub:
            repo.push_to_hub(
                commit_message="End of training", blocking=False, auto_lfs_prune=True
            )


def main(args):
    logging_dir = Path(args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = (
                torch.float16 if accelerator.device.type == "cuda" else torch.float32
            )
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=args.sample_batch_size
            )

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(
                    prompt=example["prompt"],
                    width=args.resolution,
                    height=args.resolution,
                    # negative_prompt=[args.negative_prompt_for_train]*len(example["prompt"]),
                    num_inference_steps=30,
                ).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = (
                        class_images_dir
                        / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    )
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token
                )
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        use_filename_as_label=args.use_filename_as_label,
        use_txt_as_label=args.use_txt_as_label,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    if args.use_xformers:
        print("enable xformers")
        unet.set_use_memory_efficient_attention_xformers(True)
        vae.set_use_memory_efficient_attention_xformers(True)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # optimizer
    params_to_optimize = [
        {"params": itertools.chain(unet.parameters()), "lr": args.learning_rate}
        if args.train_unet
        else None,
        {
            "params": itertools.chain(text_encoder.parameters()),
            "lr": args.learning_rate_text,
        }
        if args.train_text_encoder
        else None,
        {"params": itertools.chain(vae.parameters()), "lr": args.learning_rate_vae}
        if args.train_vae
        else None,
    ]
    params_to_optimize = [p for p in params_to_optimize if p is not None]

    # lora optimizer
    if args.use_lora:
        # freeze unet parameters
        unet.requires_grad_(False)

        # inject lora to unet, and return injected unet, lora parameters,
        # use lora parameters to optimize
        unet_lora_params, _ = inject_trainable_lora(
            unet, r=args.lora_rank, loras=args.resume_unet
        )
        params_to_optimize = (
            {"params": itertools.chain(*unet_lora_params), "lr": args.learning_rate},
        )

        if args.train_text_encoder:
            text_encoder_lora_params, _ = inject_trainable_lora(
                text_encoder,
                target_replace_module=["CLIPAttention"],
                r=args.lora_rank,
                loras=args.resume_text_encoder,
            )

            params_to_optimize = (
                {
                    "params": itertools.chain(*unet_lora_params),
                    "lr": args.learning_rate,
                },
                {
                    "params": itertools.chain(*text_encoder_lora_params),
                    "lr": args.learning_rate_text,
                },
            )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Use VeLO for faster training
    if args.use_velo:
        try:
            from pytorch_velo import VeLO
        except ImportError:
            raise ImportError(
                "To use VeLO, please install the pytorch-velo library: `python3 -m pip install git+https://github.com/janEbert/PyTorch-VeLO.git`."
            )

        num_training_steps = args.num_train_epochs * len(train_dataloader)
        optimizer = VeLO(
            params_to_optimize, num_training_steps=num_training_steps, weight_decay=0.0
        )

    else:
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,  # params_to_optimize's lr will overwrite this
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # prepare model for mixed precision training
    if not args.train_text_encoder and not args.train_vae:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    elif not args.train_vae:
        (
            unet,
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        (
            unet,
            text_encoder,
            vae,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet, text_encoder, vae, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        if args.train_vae:
            vae.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                text_embedding = text_encoder(batch["input_ids"])[0]
                # # negative text embedding
                # uncond_input = tokenizer(
                #     args.negative_prompt_for_train,
                #     padding="do_not_pad",
                #     truncation=True,
                #     max_length=tokenizer.model_max_length,
                # ).input_ids
                # uncond_input_ids = tokenizer.pad({"input_ids": [uncond_input]}, padding=True, return_tensors="pt").input_ids
                # negative_text_embedding = text_encoder(uncond_input_ids.to("cuda"))[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, text_embedding).sample

                guidance_scale = 1
                if args.with_prior_preservation:
                    # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                    noise, noise_prior = torch.chunk(noise, 2, dim=0)

                    # noise_pred_uncond = unet(noisy_latents, timesteps, negative_text_embedding).sample
                    # noise_pred_negative, noise_pred_prior_negative = torch.chunk(noise_pred_uncond, 2, dim=0)

                    # noise_pred = noise_pred_negative + guidance_scale * (noise_pred_text - noise_pred_negative)
                    # noise_pred_prior = noise_pred_prior_negative + guidance_scale * (noise_pred_prior_text - noise_pred_prior_negative)

                    # Compute instance loss
                    loss = (
                        F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                        .mean([1, 2, 3])
                        .mean()
                    )

                    # Compute prior loss
                    prior_loss = F.mse_loss(
                        noise_pred_prior.float(), noise_prior.float(), reduction="mean"
                    )

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    # noise_pred_uncond = unet(noisy_latents, timesteps, negative_text_embedding).sample
                    # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
                    loss = F.mse_loss(
                        noise_pred.float(), noise.float(), reduction="mean"
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if (
                args.save_model_every_n_epochs != None
                and (epoch % args.save_model_every_n_epochs) == 0
                and step == 0
            ):
                save_model(accelerator, unet, text_encoder, args, epoch, step)

        accelerator.wait_for_everyone()

    save_model(accelerator, unet, text_encoder, args, epoch, step)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
