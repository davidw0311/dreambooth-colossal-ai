HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DIFFUSERS_OFFLINE=1


export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./datasets/capy/instance"
export CLASS_DIR="./datasets/capy/class"
export OUTPUT_DIR="./weight_output_colossal_ai_capy_with_prior"

torchrun --nproc_per_node 1 --standalone train_dreambooth_colossalai.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=0.8 \
  --instance_prompt="a photo of @$$WaG capybara" \
  --class_prompt="a photo of capybara" \
  --resolution=512 \
  --plugin="torch_ddp_fp16" \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
