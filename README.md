# dreambooth-colossal-ai

# This repository aims to train a [DreamBooth](https://arxiv.org/abs/2208.12242) model using techniques from [Colossal AI](https://github.com/hpcaitech/ColossalAI/tree/main)

Most of the code has been copied and adapted from [https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/dreambooth](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/dreambooth)


## Instructions for installation 

On a device with cuda 11.8 and GPU, and conda installed, clone this repo then run

```
conda env create -f environment.yml
```

To activate the environment:

```
conda activate dreambooth-colossal
```

## Preparing a dataset

An example dataset is provided under [datasets/capy/instance](datasets/capy/instance)

To create your own custom dataset, upload 3-5 images of a certain subject.

## Running example using colossal-ai for dreambooth

### First edit the configurations in [colossalai.sh](colossalai.sh)

- By default we are finetuning from the [stable-diffusion-v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) model from huggingface
- Change the name to the dataset if needed

- Change the name in the prompt to a custom name if needed

```
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DIFFUSERS_OFFLINE=1

export MODEL_NAME="runwayml/stable-diffusion-v1-5" //local path or model huggingface id
export INSTANCE_DIR="./datasets/capy/instance" // path to dataset
export OUTPUT_DIR="./weight_output_colossal_ai_capy_noprior" // path to save finetuned model

torchrun --nproc_per_node 1 --standalone train_dreambooth_colossalai.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of xyzcccapy" \ //name given to this subject
  --resolution=512 \
  --plugin="low_level_zero" \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400
```

- plugin can be chosen between "torch_ddp", "torch_ddp_fp16", "gemini", and "low_level_zero", see colossal-ai documentation for more details
- gemini option will allow running with reduced GPU memory

### To run

```
bash colossalai.sh
```

This will first download the pretrained weights from huggingface. The finetuned model's unet weights should be saved under $OUTPUT_DIR

## Performing Inference
