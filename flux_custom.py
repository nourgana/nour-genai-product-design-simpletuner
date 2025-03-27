import torch
from diffusers import FluxPipeline
import PIL 

lora_model = "output/models/checkpoint-4500"

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
#pipe.load_lora_weights(lora_model)

prompt = "Front view of a Hublot watch. Titanium case with a brushed finish, ceramic bezel with a blue and black gradient representing the depths of a swimming pool. Automatic movement. Blue rubber strap with a textured pattern mimicking water ripples. Open-worked dial showcasing the movement, with blue accents. Applied baton indexes with luminescent coating. Faceted, luminescent-filled hands. Inscription 'LEON MARCHAND LIMITED EDITION' on the dial. Realistic design"

out = pipe(
    prompt=prompt,
    guidance_scale=3.5, #default 3.5
    height=768,
    width=768,
    num_inference_steps=25,
    max_sequence_length=512,
    generator=torch.manual_seed(42)
).images[0]
out.save("leon_marchand_pretrained.png")