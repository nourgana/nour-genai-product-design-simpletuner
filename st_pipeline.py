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

def generate_images(prompt, seed=42, custom=True):
    torch.cuda.empty_cache()
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    #pipe.enable_model_cpu_offload()
    if custom:
        pipe.load_lora_weights(lora_model)
    pipe.to("cuda")

    torch.cuda.empty_cache()

    images = pipe(
        prompt, 
        num_inference_steps=25, 
        guidance_scale=5.0,
        generator=torch.Generator("cpu").manual_seed(seed),
        max_sequence_length=512,
        num_images_per_prompt=2
    ).images
     
    del pipe
    return images

vertexai_project_id = "grp-prd-lvmhai-ngna1"
vertexai_location = "europe-west4"
with open("gcp_credentials.json", 'r') as json_file:
    gcp_credentials_info = json.load(json_file)

gcp_credentials = service_account.Credentials.from_service_account_info(gcp_credentials_info)

vertexai.init(project=vertexai_project_id,
              location=vertexai_location,
              credentials=gcp_credentials)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
response_schema = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string"
        },
        "symbolique_edition_limitee": {
            "type": "string"
        }
    },
    "required": ["description", "symbolique_edition_limitee"]
}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
gemini = GenerativeModel("gemini-2.0-flash",
                                   system_instruction=["You are a highly creative and experienced watch designer specializing in limited edition luxury watches. Your task is to imagine and describe an original visual design for a new Hublot limited edition watch"],

                                   generation_config = GenerationConfig(
                                       response_mime_type="application/json",
                                  response_schema=response_schema,
                                       temperature=0.2,
                                        top_p=1.0,
                                        top_k=32,
                                        candidate_count=1,
                                        max_output_tokens=8192),
                                  safety_settings = {

    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH})


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE


from google import genai
client = genai.Client(vertexai=True, project=vertexai_project_id, location=vertexai_location)

def imagen3(prompt, number_of_images=1):
    response = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt=prompt,
    config=types.GenerateImagesConfig(
        number_of_images= number_of_images,
    ))
    output = []
    for generated_image in response.generated_images:
        image = PIL.Image.open(BytesIO(generated_image.image.image_bytes)) #.save("imagen3.png")
        output.append(image)
    return output


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

def generate_prompt(ambassador, theme, movement):
    
    prompt = f"""
    
    You will be provided with the following information:

    Ambassador: {ambassador}
    Theme: {theme}
    Movement Type: {movement}

    Instructions:

    Proceed in two steps:

    STEP 1. Visual Description:

    Imagine and describe the visual design of the Hublot limited edition watch created in collaboration with [ambassador] as part of a [theme] partnership. 
    It should be a unique timepiece with an exclusive design representing the essence and identity of the collaborating entity. 
    Incorporate metaphors, subtle representations, or concepts in the design that resonate with the collaboration. Choose colors accordingly.
    Be creative but consistent. The design should make sense with the ambassador and the theme of the partnership.

    Describe the following elements in this order: Case, Bezel, Movement type: [movement], Strap, Dial, Indexes, Hands, Text inscriptions.

    Keep the description short but concise, keyword-based, and straight to the point. No bullet points.

    STEP 2. Symbolism:

    Explain the symbolism of this collaboration and your design choices. Why did you choose this specific design to represent the ambassador and theme?

    Constraints:

    * NO portraits allowed.
    * Refrain from describing elements in non-seen areas of the watch, like the case back.


    Examples:

    Collaboration with ARTURO FUENTE:
    Shiny King Gold case set with white brilliant-cut diamonds and polished King Gold bezel set with white brilliant-cut diamonds. Automatic chronograph movement. Chocolate calfskin leather strap. Brown dial. Baton indexes. Red glaive hands. Inscription "Forbidden"

    Collaboration with FIFA WORLD CUP:
    Front view of a Hublot watch. Case in polished satin-finished gold and bezel in polished satin-finished gold. Automatic movement. Black gummy alligator strap. King gold dial with the world cup embossed. Stick indexes. Sword-shaped hands.

    Collaboration with Takashi Murakami:
    Satin-finished polished black ceramic case, satin-finished polished black ceramic bezel. Automatic Unico Manufacture movement. Lined black rubber strap. Paved dial featuring features a Takashi Murakami smiling flower motif made of blue and multicolored gemstones. Sword hands.

    """
    return prompt
    
# Main app function
def main():
    st.title("Product Design Workflow MVP")

    # Initialize session state to store values across re-runs
    if 'generated_descriptions' not in st.session_state:
        st.session_state['generated_descriptions'] = ""
    if 'symbolic' not in st.session_state:
        st.session_state['symbolic'] = ""
    if 'edited_text' not in st.session_state:
        st.session_state['edited_text'] = ""
    if 'images' not in st.session_state:
        st.session_state['images'] = []
    if 'ambassador' not in st.session_state:
        st.session_state['ambassador'] = ""
    if 'theme' not in st.session_state:
        st.session_state['theme'] = ""

    # Step 1: Ask for user inputs
    st.session_state['ambassador'] = st.text_input("Enter the name of the ambassador:", 
                                                   value=st.session_state['ambassador'],
                                                   placeholder="for example, Leon Marchand")
    st.session_state['theme'] = st.text_input("Enter the theme of the collaboration:", 
                                              value=st.session_state['theme'],
                                              placeholder="for example, Sports")
    
    movement = st.selectbox("Select a movement", options=["Automatic movement",
        "Hand-wound movement",
        "Self-winding movement",
        "Self-winding chronograph movement",
        "Self-winding movement with moon phase",
        "Quartz movement",
        "Self-winding Unico movement",
        "Automatic winding movement",
        "Skeleton tourbillon manual-winding movement"
        ],
        index=None,
        placeholder="Choose from the list below...",
   )

    if movement and st.button("Generate Text"):
        st.write("You selected:", movement)
        if st.session_state['ambassador'] and st.session_state['theme']:
            # Step 2: Process the inputs and generate text
            prompt = generate_prompt(st.session_state['ambassador'], st.session_state['theme'], movement=movement)
            #Simulate gemini.generate_content for test
            output = ast.literal_eval(gemini.generate_content([prompt]).text)

            st.session_state['generated_descriptions'] = output["description"]
            st.session_state['symbolic'] = output["symbolique_edition_limitee"]

        else:
            st.error("Please enter both the ambassador and the theme.")

    if st.session_state['generated_descriptions']:  # Only show if text has been generated
        st.write("Generated Text:")
        st.write(st.session_state['generated_descriptions'])

        # Allow the user to edit the generated text
        st.session_state['edited_text'] = st.text_area("Edit the text:", st.session_state['generated_descriptions'])
        st.session_state['edited_text'] = "Front view of a Hublot watch. " + st.session_state['edited_text']

        # Validate the edited text with a button
        if st.button("Validate Text") :
            st.session_state['num_images'] = st.slider("Choose the number of images to generate:", min_value=0, value=2, max_value=10)
            #num_images = st.number_input("Choose the number of images to generate", value=None)
            if st.session_state['num_images']>0:
                with st.spinner(f'Generating images...'):
                    st.session_state['images'] = generate_images(st.session_state['edited_text'] + "High Realism and quality.") 
                    
        with open(f"streamlit_app/{st.session_state['ambassador']}_{st.session_state['theme']}.txt", "w") as f:
            f.write(st.session_state['edited_text'])
            f.close()

        if st.session_state['images']: #Only execute after validation
            st.write("Final Text:")
            st.write(st.session_state['edited_text'])
            
            #st.session_state['images'].append(imagen3(prompt=st.session_state['edited_text'])[0])

            # Step 3: Return images in a row
            cols = st.columns(st.session_state['num_images'])
            
            #scores = compute_clip_scores(images=st.session_state['images'], text=st.session_state['edited_text'])
            
            for i, col in enumerate(cols):
                with col:
                    #caption = f"Example {i+1} - CLIP score : {scores[i][0]}"
                    caption = f"Example {i+1}"
                    if i==2:
                        caption="imagen3"
                    st.image(st.session_state['images'][i], caption=caption, use_container_width=True)
                    st.session_state['images'][i].save(f"streamlit_app/{st.session_state['ambassador']}_{st.session_state['theme']}_{caption}.jpg")
                    
            #combine_images_horizontally(st.session_state['images'], f"streamlit_app/{st.session_state['ambassador']}_{st.session_state['theme']}_combined.jpg",
            #                            ["Example 1", "Example 2", "Imagen 3"])

            st.write("Explainations:")
            st.write(st.session_state['symbolic'])
            st.write()

if __name__ == "__main__":
    main()