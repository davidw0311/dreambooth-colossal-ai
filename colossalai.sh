HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DIFFUSERS_OFFLINE=1

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./datasets/capy/instance"
export OUTPUT_DIR="./weight_output_colossal_ai_capy_noprior"

torchrun --nproc_per_node 1 --standalone train_dreambooth_colossalai.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of xyzcccapy" \
  --resolution=512 \
  --plugin="low_level_zero" \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400
