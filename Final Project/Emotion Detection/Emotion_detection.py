import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image

# Define class labels for emotions
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load the ResNet50 model
@st.cache_resource
def load_model(model_path):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(EMOTION_LABELS))  # Adjust output layer
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# Image transformations (matching training preprocessing)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.Resize((224, 224)),  # Resize to ResNet50 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])


# Function to enhance contrast before face detection
def enhance_contrast(image):
    # Check if the image is grayscale (single channel)
    if len(image.shape) == 2:
        gray = image  # Already grayscale
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed

    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    return equalized


# Function to detect faces using OpenCV
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Enhance contrast in the image
    enhanced_image = enhance_contrast(image)

    # Detect faces in the enhanced grayscale image
    faces = face_cascade.detectMultiScale(enhanced_image, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    return faces


# Function to predict emotion with confidence score
def predict_emotion(model, face_img):
    face_img = Image.fromarray(face_img).convert("RGB")  # Convert to RGB
    face_tensor = transform(face_img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(face_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Convert to probabilities
        confidence, predicted = torch.max(probabilities, 1)

    predicted_emotion = EMOTION_LABELS[predicted.item()]
    return predicted_emotion, confidence.item()


# Streamlit UI
st.title("😊 Emotion Detection from Images")
st.write("Upload an image, and the model will detect faces and classify emotions.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)  # Convert PIL image to NumPy array

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Load model
    model_path = r"C:\Users\Mohan\PycharmProjects\pythonProject\Final project\emotion_resnet50.pth"
    model = load_model(model_path)

    # Detect faces
    faces = detect_faces(image)

    if len(faces) == 0:
        st.warning("No faces detected. Please upload another image.")
    else:
        # Confidence threshold
        threshold = 0.4  # Lower threshold to allow for more visible labels

        # Process each detected face
        for (x, y, w, h) in faces:
            face_img = image[y:y + h, x:x + w]  # Crop face region
            face_img = cv2.resize(face_img, (224, 224))  # Resize face before prediction

            emotion, confidence = predict_emotion(model, face_img)  # Predict emotion

            # Debugging - Display confidence values
            st.write(f"Predicted: {emotion}, Confidence: {confidence:.2f}")

            # Set color for bounding box based on emotion
            emotion_colors = {
                "Surprise": (0, 255, 255),  # Yellow
                "Angry": (0, 0, 255),  # Red
                "Happy": (0, 255, 0),  # Green
                "Sad": (255, 0, 0),  # Blue
                "Neutral": (255, 255, 0),  # Cyan
                "Fear": (255, 0, 255),  # Magenta
                "Disgust": (255, 165, 0),  # Orange
            }
            emotion_color = emotion_colors.get(emotion, (255, 255, 255))  # Default to white

            # Ensure label is positioned above the bounding box and does not overlap
            label_position = (x, max(y - 10, 20))  # Prevent label from going off image

            # If confidence is above threshold, show the label
            if confidence >= threshold:
                label_text = f"{emotion} ({confidence:.2f})"
                cv2.putText(image, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)

            # Draw bounding box with emotion color
            cv2.rectangle(image, (x, y), (x + w, y + h), emotion_color, 2)

        # Display result
        st.image(image, caption="Emotion Detected", use_container_width=True)
