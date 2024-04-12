from accelerate import Accelerator
from diffusers import DiffusionPipeline
import argparse
import os
import numpy as np

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a inference script")
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet_checkpoint",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained finetuned unet model (folder containing diffusion_pytorch_model.bin)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        required=True,
        help="prompt for text to image generation",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="prompt for dreambooth specific text to image generation",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        required=False,
        help="guidance scale",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=50,
        required=False,
        help="number of denoising steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_output",
        required=False,
        help="path of inference output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        required=False,
        help="random seed for generation",
    )


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    
    return args


args = parse_args()

seed = args.seed
if seed is None:
    seed = np.random.randint(0, np.iinfo(np.int16).max + 1)
    
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

prompt = args.prompt
instance_prompt = args.instance_prompt
print(f'generating image of "{prompt}" using random seed: {seed}')
# Load the pipeline 
model_id = args.model_id
pipeline = DiffusionPipeline.from_pretrained(model_id, safety_checker=None).to("cuda")

print("performing inference with original model")
image = pipeline(prompt, num_inference_steps=args.inference_steps, guidance_scale=args.guidance_scale).images[0]
image.save(f"{args.output_dir}/original_output.png")


accelerator = Accelerator()

# load the dreambooth finetuned unet weights
unet = accelerator.prepare(pipeline.unet)

# Restore state from a checkpoint path. You have to use the absolute path here.
accelerator.load_state(args.unet_checkpoint)

# Rebuild the pipeline with the unwrapped models (assignment to .unet and .text_encoder should work too)
pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    unet=accelerator.unwrap_model(unet),
    safety_checker=None
).to('cuda')

print("performing inference with dreambooth model")
dreambooth_image = pipeline(prompt, num_inference_steps=args.inference_steps, guidance_scale=args.guidance_scale).images[0]
dreambooth_image.save(f"{args.output_dir}/dreambooth_output.png")

if args.instance_prompt:
    print("performing inference with dreambooth model using instance prompt")
    dreambooth_image = pipeline(instance_prompt, num_inference_steps=args.inference_steps, guidance_scale=args.guidance_scale).images[0]
    dreambooth_image.save(f"{args.output_dir}/dreambooth_output_instance.png")


