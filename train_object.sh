# 用于训练特定物体/人物的方法（只需单一标签）
export MODEL_NAME="./base_model"
export INSTANCE_DIR="./datasets/test_arb"
export OUTPUT_DIR="./new_model"
export CLASS_DIR="./datasets/class" # 用于存放模型生成的先验知识的图片文件夹，请勿改动
export LOG_DIR="/root/tf-logs"
export TEST_PROMPTS_FILE="./test_prompts_object.txt"

rm -rf $CLASS_DIR/* # 如果你要训练与上次不同的特定物体/人物，需要先清空该文件夹。其他时候可以注释掉这一行（前面加#）
rm -rf $LOG_DIR/*

# 以下两行是用于训练文本编码器的，默认不启用
# --train_text_encoder \
# --learning_rate_text=1e-5 \
  
accelerate launch tools/train_dreambooth.py \
  --use_lora \
  --lora_rank=128 \
  --resolution=512 \
  --train_unet \
  --learning_rate=1e-4 \
  --use_xformers \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_prompt="sks boy" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_prompt="boy" \
  --class_data_dir=$CLASS_DIR \
  --num_class_images=200 \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --logging_dir=$LOG_DIR \
  --mixed_precision="fp16" \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --set_grads_to_none \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=0 \
  --auto_test_model \
  --test_prompts_file=$TEST_PROMPTS_FILE \
  --negative_prompt_for_test="low quality, worst quality, bad anatomy, inaccurate limb, bad composition, inaccurate eyes, extra digit,fewer digits, extra arms" \
  --test_width=512 \
  --test_height=512 \
  --test_seed=1212123 \
  --test_num_per_prompt=2 \
  --num_train_epochs=10 \
  --save_model_every_n_epochs=2 
  
# 如果num_train_epochs改大了，请记得把save_model_every_n_epochs也改大，不然磁盘容易中间就满了
# --use_lora 是否使用lora方法训练，删除该行可使用默认的Dreambooth
# 使用lora时默认不保存完整模型，只存lora权重部分。

# --train_text_encoder 训练文本编码器

# lora相关：
# --lora_rank (down层的输出尺寸、也是up层的输入尺寸)，越大网络越复杂，看情况自己设定
# --resume_unet --resume_text_encoder 可用于训练中断时恢复训练
# 如--resume_unet="path/xxx.pt" path是路径的意思 xxx是文件名
# --resume_text_encoder="path/xxx.text_encoder.pt" 如果也训练了text encoder的话就带上这个

# 以下是核心参数介绍：
# 主要的几个
# --train_text_encoder 训练文本编码器
# --mixed_precision="fp16" 混合精度训练
# - resolution 
# 图片的分辨率，设置为768会自动按比缩放输入图像的短边到768
# - instance_prompt
# 如果你希望训练的是特定的人物，使用该参数
# 如 --instance_prompt="a photo of <xxx> girl"
# - use_txt_as_label
# 是否读取与图片同名的txt文件作为label
# 如果你要训练的是整个大模型的图像风格，那么可以使用该参数
# 该选项会忽略instance_prompt参数传入的内容
# - learning_rate
# 学习率，一般是2e-6，是训练中需要调整的关键参数
# 太大会导致模型不收敛，太小的话，训练速度会变慢
# - max_train_steps
# 训练的最大步数，一般是1000，如果你的数据集比较大，那么可以适当增大该值
# - save_model_every_n_epochs
# 每多少轮保存一次模型，方便查看中间训练的结果找出最优的模型，也可以用于断点续训

# --train_text_encoder # 除了图像生成器，也训练文本编码器

# --auto_test_model, --test_prompts_file, --test_seed, --test_num_per_prompt
# 分别是自动测试模型（每save_model_every_n_steps步后）、测试的文本、随机种子、每个文本测试的次数
# 测试的样本图片会保存在模型输出目录下的test文件夹中

# --lr_scheduler 学习率策略：
# "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"