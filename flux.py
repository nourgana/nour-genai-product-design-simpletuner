import streamlit as st
from diffusers import FluxPipeline
import torch
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory, Part, Image
)
import vertexai.generative_models as genai
import json
import ast 
import PIL 
from google import genai
from google.genai import types
from io import BytesIO
import os 
from utils import combine_images_horizontally
from clip_score import compute_clip_scores

lora_model = "output/models/checkpoint-4500"
def generate_images(prompt, seed, custom=True):
    torch.cuda.empty_cache()
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    if custom:
        pipe.load_lora_weights(lora_model)
    #pipe.to("cuda")

    torch.cuda.empty_cache()

    pipeline_output = pipe(
        prompt, 
        num_inference_steps=24, 
        guidance_scale=5.0,
        generator=torch.Generator("cpu").manual_seed(seed),
        max_sequence_length=512,
        num_images_per_prompt=2
    )
    
    del pipe
    return pipeline_output

images = generate_images("boy in LA", seed=42).images
images[0].save("0.png")
images[1].save("1.png")
