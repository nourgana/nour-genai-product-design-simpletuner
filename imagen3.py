from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import json 
import os 
from google.oauth2 import service_account

vertexai_project_id = "grp-prd-lvmhai-ngna1"
vertexai_location = "europe-west4"
with open("gcp_credentials.json", 'r') as json_file:
    gcp_credentials_info = json.load(json_file)

gcp_credentials = service_account.Credentials.from_service_account_info(gcp_credentials_info)


client = genai.Client(vertexai=True, project=vertexai_project_id, location=vertexai_location)#, credentials=gcp_credentials)

response = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt='Fuzzy bunnies in my kitchen',
    config=types.GenerateImagesConfig(
        number_of_images= 1,
    )
)
for generated_image in response.generated_images:
  image = Image.open(BytesIO(generated_image.image.image_bytes)).save("imagen3.png")