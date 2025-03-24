from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

def compute_clip_scores(images:list, text:str):

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(text=text, images=images, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    #probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

    #print(logits_per_image)
    
    return logits_per_image.detach().numpy()


images = [Image.open(f"streamlit_app/Donald Trump_Politics_Example 1.jpg"),
        Image.open(f"streamlit_app/Donald Trump_Politics_Example 2.jpg")]

with open(f"streamlit_app/Donald Trump_Politics.txt", "r") as f:
    text = f.read()
    f.close()
    
compute_clip_scores(images=images, text=text)