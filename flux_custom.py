import torch
from diffusers import FluxPipeline
import PIL 

lora_model = "output_LV/models_prev/checkpoint-2000"

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.load_lora_weights(lora_model)

prompt =  "a louis vuitton monogram canvas cover book with multicolored dots and an S-lock closure"
out = pipe(
    prompt=prompt,
    guidance_scale=3.5, #default 3.5
    height=768,
    width=768,
    num_inference_steps=25,
    generator=torch.manual_seed(0)
).images[0]
out.save("custom_flux_bookcover.png")