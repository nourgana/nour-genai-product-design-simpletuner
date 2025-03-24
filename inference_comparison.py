###
# This script will generate images for the same seed/prompt across many models and stitch the outputs together.
###

from diffusers import FluxPipeline
from torch import manual_seed, float16
import os
from PIL import Image, ImageDraw, ImageFont
#from helpers.prompts import prompts

prompts = {
    "525.OX.0180.LR.0904": "Front view of a Hublot watch. Shiny King Gold case set with white diamonds and polished King Gold bezel set with baguette-cut white diamonds. Automatic chronograph movement. Black alligator strap. Sapphire dial, skeleton. Stick indexes. Stick hands.",
    "507.CX.9004.RX.TAK23": "Front view of a Hublot watch. Satin-finished polished black ceramic case, satin-finished polished black ceramic bezel. Automatic Unico Manufacture movement. Lined black rubber strap. Dial paved with pink, yellow, blue, orange, violet and green sapphires, forming a smiling flower. Sword-type hands",
    "541.NO.1180.LR": "Front view of a Hublot watch. Satin-finished polished titanium case with satin-finished polished King Gold titanium bezel. Automatic chronograph movement. Black alligator strap. Matte black dial. Golden baton indexes. Golden sword-shaped hands.', '568.NX.891L.NX.1204': 'Front view of a Hublot watch. Case in polished and satin-finished titanium and polished titanium bezel set with brilliant-cut white diamonds. Automatic movement. Polished and satin-finished titanium bracelet. Sunray blue dial. Baton indexes. Polished hands",
    "568.NX.897M.NX.1204": "Front view of a Hublot watch. Case in polished and satin-finished titanium and bezel in polished titanium set with brilliant-cut white diamonds. Automatic movement. Polished and satin-finished titanium bracelet. Brown dial set with brilliant-cut white diamonds. Diamond indexes. Polished sword-shaped hands",
    "505.CS.1270.VR": "Front view of a Hublot watch. Polished black ceramic case with polished black ceramic bezel. Skeleton tourbillon manual-winding manufacture movement. Glossy black calfskin strap. Glossy black dial. Black baguette diamond indexes. Polished sword-shaped hands",
    "585.OX.898P.OX.1204": "Front view of a Hublot watch. Satin-finished and polished King Gold case with polished King Gold bezel set with brilliant white diamonds. Automatic movement. KING GOLD PINK DIAMONDS bracelet. Pink dial set with brilliant white diamonds. Diamond indexes. Polished sword-shaped hands.",
    "542.OX.7180.RX": "Front view of a Hublot watch. Polished and satin-finished King Gold case and polished satin-finished King Gold bezel. Automatic movement. Blue lined rubber strap. Sunray blue dial. Stick indexes. Sword-shaped hands.', '511.OX.7180.LR': 'Front view of a Hublot watch. Case in polished and satin-finished King Gold with polished and satin-finished King Gold bezel. Automatic movement. ALL BLUE ALLIGATOR strap. Sunray blue dial. Stick indexes. Polished hands"

}

# Define your pipelines and settings in a list of dictionaries
pipelines_info = [
    {
        "label": "base_flux",
        "pretrained_model": "black-forest-labs/FLUX.1-dev",
        "settings": {
            "guidance_scale": 3.5,
            #"guidance_rescale": 0.7,
            "num_inference_steps": 25,
        },
    },
    {"label": "lora_flux",
     "pretrained_model": "black-forest-labs/FLUX.1-dev",
     "lora": {"model": "output/models/checkpoint-4500"},
     "settings": {"guidance_scale": 3.5, 
                  #"guidance_rescale": 0.7,
                  "num_inference_steps": 25, 
                  }
     }
]


def combine_and_label_images(images_info, output_path):
    # Assume images_info is a list of tuples: (Image object, label)
    # Initial setup based on the first image dimensions and number of images
    label_height = 45
    total_width = sum(image.width for image, _ in images_info)
    max_height = max(image.height for image, _ in images_info) + label_height
    combined_image = Image.new("RGB", (total_width, max_height), "white")

    # Combine images and labels
    current_x = 0
    for image, label in images_info:
        combined_image.paste(image, (current_x, label_height))
        current_x += image.width

    # Adding labels using a uniform method for text placement
    draw = ImageDraw.Draw(combined_image)
    try:
        # Attempt to use a specific font
        font = ImageFont.truetype(
            ".venv/lib/python3.11/site-packages/cv2/qt/fonts/DejaVuSans.ttf", 40
        )  # Adjust font path and size
    except IOError:
        # Fallback to default font
        font = ImageFont.load_default()

    current_x = 0
    for _, label in images_info:
        draw.text((current_x + 10, 2), label, fill="black", font=font)
        current_x += image.width

    combined_image.save(output_path)

for shortname, prompt in prompts.items():
    print(f"Processing: {shortname}")
    target_dir = f"inference/images"
    os.makedirs(target_dir, exist_ok=True)

    images_info = []
    for pipeline_info in pipelines_info:
        images_info.append((Image.open(f"test_folder/{shortname}.jpg"), "target watch model"))
        # Initialize pipeline
        pipeline = FluxPipeline.from_pretrained(
            pipeline_info["pretrained_model"],
            torch_dtype=float16,
        ).to("cuda")
        pipeline.enable_model_cpu_offload()

        # Load LoRA weights if specified
        if "lora" in pipeline_info:
            pipeline.load_lora_weights(
                pipeline_info["lora"]["model"],
                #weight_name=pipeline_info["lora"]["weight_name"],
            )

        # Generate image with specified settings
        settings = pipeline_info.get("settings", {})
        image = pipeline(prompt, generator=manual_seed(420420420), **settings).images[0]
        # Unload LoRA weights if they were loaded
        if "lora" in pipeline_info:
            pipeline.unload_lora_weights()
        del pipeline
        #image.save(image_path, format="PNG")

        images_info.append((image, pipeline_info["label"]))


    # Combine and label images
    print("images_info: ", images_info)
    combine_and_label_images(images_info, f"{target_dir}/{shortname}_combined_image.png")