import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Mapping of class indices to disease names
class_mapping = {
    0: "Apple___Apple_scab", 1: "Apple___Black_rot", 2: "Apple___Cedar_apple_rust", 3: "Apple___healthy",
    4: "Blueberry___healthy", 5: "Cherry_(including_sour)___Powdery_mildew", 6: "Cherry_(including_sour)___healthy",
    7: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", 8: "Corn_(maize)___Common_rust_",
    9: "Corn_(maize)___Northern_Leaf_Blight", 10: "Corn_(maize)___healthy", 11: "Grape___Black_rot",
    12: "Grape___Esca_(Black_Measles)", 13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 14: "Grape___healthy",
    15: "Orange___Haunglongbing_(Citrus_greening)", 16: "Peach___Bacterial_spot", 17: "Peach___healthy",
    18: "Pepper,_bell___Bacterial_spot", 19: "Pepper,_bell___healthy", 20: "Potato___Early_blight",
    21: "Potato___Late_blight", 22: "Potato___healthy", 23: "Raspberry___healthy", 24: "Soybean___healthy",
    25: "Squash___Powdery_mildew", 26: "Strawberry___Leaf_scorch", 27: "Strawberry___healthy",
    28: "Tomato___Bacterial_spot", 29: "Tomato___Early_blight", 30: "Tomato___Late_blight", 31: "Tomato___Leaf_Mold",
    32: "Tomato___Septoria_leaf_spot", 33: "Tomato___Spider_mites Two-spotted_spider_mite",
    34: "Tomato___Target_Spot", 35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus", 36: "Tomato___Tomato_mosaic_virus",
    37: "Tomato___healthy"
}

# Load the trained ResNet-50 model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)  # Don't load default weights
    num_ftrs = model.fc.in_features  # Get input size of final layer
    model.fc = torch.nn.Linear(num_ftrs, 38)  # Modify FC layer for 38 classes
    model.load_state_dict(torch.load(r"C:\Users\Mohan\PycharmProjects\pythonProject\Final project\plant_disease_resnet50.pth", map_location=torch.device("cpu")))
    model.eval()  # Set model to evaluation mode
    return model

# Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet-50 requires 224x224 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ResNet-50 normalization
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Disease Prediction Function
def predict_disease(image):
    model = load_model()
    img_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return class_mapping.get(predicted_class, "Unknown")

# Streamlit UI
st.set_page_config(page_title="🌿 Plant Disease Detection", layout="centered")

st.title("🌿 Plant Disease Detection using ResNet-50")
st.write("Upload a plant leaf image to detect potential diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB format
    st.image(image, caption="Uploaded Image", use_container_width=True)


    if st.button("Predict Disease"):
        prediction = predict_disease(image)
        st.success(f"Prediction: {prediction}")
