
from diffusers import StableDiffusionPipeline
import torch
from diffusers import DDIMScheduler

model_path = "./output_merged"  
prompt = "a sks boy, very old"
negative_prompt="low quality, worst quality, bad anatomy, inaccurate limb, bad composition, inaccurate eyes, extra digit,fewer digits, extra arms"

torch.manual_seed(123123123)

pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        scheduler=DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=True,
            steps_offset=1
        ),
        safety_checker=None
    )

pipe = pipe.to("cuda")
images = pipe(
    prompt=prompt,
    width=512,
    height=768,
    negative_prompt=negative_prompt,
    num_inference_steps=30, 
    num_images_per_prompt=3,
).images
for i, image in enumerate(images):
    image.save(f"test-{i}.png")