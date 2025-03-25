import torch
from diffusers import FluxPipeline
import PIL 

lora_model = "output/models/checkpoint-4500"

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
#pipe.load_lora_weights(lora_model)

prompt = "Front view of a Hublot Watch. Polished titanium case, rainbow-set sapphire crystal bezel. Automatic movement. Light blue alligator strap with rainbow stitching. Mother-of-pearl dial with signature 'Flirting' eye logo at the center. Applique indexes. Faceted hands. 'Hublot Loves Chiara Ferragni' inscription."
out = pipe(
    prompt=prompt,
    guidance_scale=3.5, #default 3.5
    height=768,
    width=768,
    num_inference_steps=25,
    generator=torch.manual_seed(42)
).images[0]
out.save("chiara_ferragni.png")