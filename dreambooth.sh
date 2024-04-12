python train_dreambooth.py \
   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  \
  --instance_data_dir="./datasets/capy/instance" \
  --output_dir="./weight_output_dreambooth_capy" \
  --instance_prompt="a picture of a xyzcccapy" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 
