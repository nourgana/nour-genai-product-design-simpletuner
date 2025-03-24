import torch
from diffusers import FluxPipeline
import PIL 

lora_model = "output/models/checkpoint-4500"

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.load_lora_weights(lora_model)

prompt = "Front view of a Hublot watch. Satin-finished polished black ceramic case, satin-finished polished black ceramic bezel. Automatic Unico Manufacture movement. Lined black rubber strap. Dial paved with pink, yellow, blue, orange, violet and green sapphires, forming a smiling flower. Sword-type hands."
out = pipe(
    prompt=prompt,
    guidance_scale=3.5, #default 3.5
    height=768,
    width=768,
    num_inference_steps=25,
    generator=torch.manual_seed(42)
).images[0]
out.save("lora_507.CX.9004.RX.TAK23.png")