import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import requests
import os
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ✅ Define your class names manually or load from training
class_names = sorted(os.listdir("~path/working/data/train"))  # Or hardcode list

# ✅ Load MobileNetV2 model and weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load("mobilenetv2_plant_disease.pth", map_location=device)) 
model.eval().to(device)

# ✅ Image preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess_image(image):
    image = image.convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

def predict(image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)
        return class_names[predicted.item()], probs[0][predicted.item()].item() * 100

# ✅ Disease info API (unchanged)
def get_disease_info(label):
    url = "https://api.hyperbolic.xyz/v1/chat/completions" # Replace with your API endpoint
    headers = {
        "Content-Type": "application/json",
        "Authorization": "API-KEY"
    }
    prompt = ( 
        f"You are a plant disease expert. The label is '{{label}}'. Always reply in this exact format and nothing else.\n"
        f"If the label describes a healthy leaf, reply: 'healthy [plant name]' on the first line in bold letters, then on the next line, provide a brief description of what a healthy leaf of this plant looks like.\n"
        f"If the label describes a disease, reply: '[disease] [plant name]' on the first line in bold letters, then on the next line, provide a brief description of what a leaf with this disease looks like. Then add:\n"
        f"Symptoms:\n- point 1\n- point 2 (max 3, compress if needed)\n"
        f"Cure:n- point 1\n- point 2 (max 3, compress if needed)\n"
        f"Do not add any extra explanation, only reply in this format.\n"
        f"Examples:\n"
        f"Input: Tomato LateBlight\nOutput: lateblight tomato\n- Late blight on tomato appears as dark, greasy-looking lesions on leaves, stems, and fruit, often surrounded by pale green or yellow halos. Infected plants deteriorate rapidly, with fruit turning brown and leathery, especially in cool, wet conditions.\nSymptoms:\n- Yellowing of leaves\n- Dark lesions on stems\nCure:\n- Remove infected leaves\n- Apply fungicide\n"
        f"Input: Aloe Vera Healthy\nOutput: healthy aloe vera\n- Healthy aloe vera has thick, fleshy green leaves with a vibrant color and firm texture, often edged with small white teeth. It grows upright in a rosette pattern and feels cool and moist to the touch when cut, indicating rich gel content.\n"
        f"Now, for this label: {label}"
        )
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "deepseek-ai/DeepSeek-R1-0528",  # Replace with your model name
        "max_tokens": 508,
        "temperature": 0.1,
        "top_p": 0.9
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Could not fetch disease information."

def format_disease_info(label, api_response):
    import re
    api_response = re.sub(r'<think>[\s\S]*?</think>', '', api_response, flags=re.IGNORECASE)
    return api_response.strip()

# ✅ Streamlit UI
st.set_page_config(page_title="Plant Disease Detector")

st.markdown(
    """
    <style>
    .centered-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    </style>
    <div class="centered-content">""",
    unsafe_allow_html=True
)

st.markdown("<h1>Plant Disease Detector</h1>", unsafe_allow_html=True)
st.markdown("### Upload your image", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    input_tensor = preprocess_image(image)
    label, accuracy = predict(input_tensor)
    st.success(f"Prediction accuracy: {accuracy:.2f}%")
    st.image(image, caption="Uploaded Image", width=300)

    with st.spinner("Fetching disease information..."):
        disease_info = get_disease_info(label)
        formatted_info = format_disease_info(label, disease_info)
    st.markdown(formatted_info)

st.markdown("</div>", unsafe_allow_html=True)
