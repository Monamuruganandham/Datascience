import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2

# Set Streamlit Page Configuration
st.set_page_config(page_title="🌿 Plant Disease Detector", page_icon="🍃", layout="wide")

# Disease Class Mapping with Symptoms
class_mapping = {
    0: ("Apple___Apple_scab", "- Olive-green to brown fuzzy spots on leaves.\n- Spots darken, causing leaves to twist, pucker, and yellow.\n- Severe infection leads to premature leaf drop and distorted fruit."),
    1: ("Apple___Black_rot", "- Dark, sunken lesions on fruit.\n- Leaf yellowing.\n- Tiny, purplish specks appear on the upper surfaces of leaves.\n- Heavily infected leaves may become chlorotic (yellow) and fall prematurely."),
    2: ("Apple___Cedar_apple_rust",
        "- Small, pale yellow spots on the upper leaf surface.\n- Spots enlarge, turning bright orange-yellow with a red border.\n- Hairlike projections (aecia) appear on the underside of leaves.\n- Severe infection causes premature leaf drop.\n- Orange spots may develop on the fruit, sometimes leading to early fruit drop."),
    3: ("Apple___healthy", "No disease detected."),
    4: ("Blueberry___healthy", "No disease detected."),
    5: ("Cherry_(including_sour)___Powdery_mildew",
        "- Light, powdery patches on young leaves.\n- Severe infections cause leaves to pucker and twist.\n- Infected leaves become smaller, pale, and distorted."),
    6: ("Cherry_(including_sour)___healthy", "No disease detected."),
    7: ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "- Small, necrotic pinpoints with yellow halos.\n- Lesions turn rectangular, brown to gray, spanning secondary veins.\n- Severe infections cause lesions to merge, killing entire leaves."),
    8: ("Corn_(maize)___Common_rust_", "- Small, oval, dark-reddish-brown pustules on leaves.\n- Pustules appear on both upper and lower leaf surfaces.\n- Rust-colored spores darken as they mature.\n- Younger leaf tissue is more susceptible, leading to chlorosis and yield loss."),
    9: ("Corn_(maize)___Northern_Leaf_Blight",
        "- Long, elliptical or cigar-shaped lesions on corn leaves.\n- Lesions start grayish-green and mature to tan or pale gray.\n- In severe cases, entire leaves become blighted, making the plant look prematurely dead or gray."),
    10: ("Corn_(maize)___healthy", "No disease detected."),
    11: ("Grape___Black_rot",
         "- Small, round, reddish-brown spots on upper leaf surface.\n- Spots start tan with a red margin, then turn brown with a dark red border.\n- Shoots and petioles show oval, dark brown lesions."),
    12: ("Grape___Esca_(Black_Measles)",
         "- Interveinal striping with angular reddish-brown lesions.\n- Lesions dry and become necrotic.\n- Red cultivars show dark red lesions; white cultivars show yellow lesions.\n- Leaves may appear yellowish, cupped, and tattered around edges."),
    13: ("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
         "- Linear reddish-brown streaks on shoots, extending from base to tip.\n- Shoots may wilt, droop, or dry up.\n- Pale yellowish-green spots on lowest internodes of young shoots."),
    14: ("Grape___healthy", "No disease detected."),
    15: ("Orange___Haunglongbing_(Citrus_greening)",
         "- Blotchy mottling with yellowing leaves.\n- Asymmetrical yellowing, differing from nutrient deficiencies.\n- Small, narrow leaves compared to healthy trees."),
    16: ("Peach___Bacterial_spot",
         "- Water-soaked lesions on leaves and fruit.\n- Small, dark spots develop and may merge.\n- Leaves turn yellow and drop prematurely."),
    17: ("Peach___healthy", "No disease detected."),
    18: ("Pepper,_bell___Bacterial_spot",
         "- Small, dark, water-soaked leaf spots.\n- Spots enlarge and become necrotic.\n- Fruit may develop raised, scabby spots."),
    19: ("Pepper,_bell___healthy", "No disease detected."),
    20: ("Potato___Early_blight",
         "- Brown leaf spots with concentric rings.\n- Older leaves affected first.\n- Leaves may wilt and die prematurely."),
    21: ("Potato___Late_blight",
         "- Dark lesions with white mold growth.\n- Rapid leaf wilting and collapse.\n- Infected tubers develop dark, firm rot."),
    22: ("Potato___healthy", "No disease detected."),
    23: ("Raspberry___healthy", "No disease detected."),
    24: ("Soybean___healthy", "No disease detected."),
    25: ("Squash___Powdery_mildew",
         "- White powdery fungal patches on leaves.\n- Leaves may yellow and curl.\n- Reduced plant vigor and fruit yield."),
    26: ("Strawberry___Leaf_scorch",
         "- Brown, dry leaf edges.\n- Small reddish-purple spots on leaves.\n- Leaves may wither and die."),
    27: ("Strawberry___healthy", "No disease detected."),
    28: ("Tomato___Bacterial_spot",
         "- Dark, water-soaked leaf and fruit spots.\n- Spots enlarge, turning black and necrotic.\n- Severe infections lead to defoliation."),
    29: ("Tomato___Early_blight",
         "- Brown leaf spots with yellow halo.\n- Concentric rings develop within spots.\n- Lower leaves affected first, leading to defoliation."),
    30: ("Tomato___Late_blight",
         "- Dark, wet lesions with mold growth.\n- Rapid leaf yellowing and collapse.\n- Fruit develops firm, dark rots."),
    31: ("Tomato___Leaf_Mold",
         "- Yellow leaf spots with fuzzy mold.\n- Spots enlarge, causing leaf curling.\n- Severe cases lead to defoliation."),
    32: ("Tomato___Septoria_leaf_spot",
         "- Small, circular brown leaf spots.\n- Spots have dark borders and light centers.\n- Leaves may yellow and fall off."),
    33: ("Tomato___Spider_mites Two-spotted_spider_mite",
         "- Fine webbing and yellowed leaves.\n- Leaves develop stippling and curl.\n- Heavy infestations cause leaf drop."),
    34: ("Tomato___Target_Spot",
         "- Dark leaf lesions with concentric rings.\n- Lesions enlarge and merge.\n- Infected leaves yellow and drop."),
    35: ("Tomato___Tomato_Yellow_Leaf_Curl_Virus",
         "- Yellow, curled leaves; stunted growth.\n- Leaves become thick and brittle.\n- Reduced fruit production."),
    36: ("Tomato___Tomato_mosaic_virus",
         "- Mottled, discolored leaves; stunted growth.\n- Leaves develop necrotic streaks.\n- Fruit may be deformed or discolored."),
    37: ("Tomato___healthy", "No disease detected.")
}


@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 38)
    model.load_state_dict(
        torch.load(r"C:\Users\Mohan\PycharmProjects\pythonProject\Final project\plant_disease_resnet50.pth",
                   map_location=torch.device("cpu")))
    model.eval()
    return model


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def detect_leaf(image):
    open_cv_image = np.array(image)
    hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2HSV)
    lower_green, upper_green = np.array([25, 40, 40]), np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return np.count_nonzero(mask) / mask.size > 0.05


def predict_disease(image):
    if not detect_leaf(image):
        return "⚠️ No leaf detected. Please upload a plant leaf image.", "error", None

    model = load_model()
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    disease, symptoms = class_mapping.get(predicted_class, ("Unknown", "No symptoms available."))
    if "healthy" in disease:
        return f"✅ {disease} (Healthy Plant)", "success",None
    else:
        return f"🚨 {disease} (Disease Detected)", "error", symptoms


st.title("🌿 Plant Disease Detector")
st.markdown("**📷 Upload a Leaf Image to Instantly Check for Plant Diseases!**")

st.header("📤 Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.sidebar.image(image, caption="📷 Uploaded Image", width=250)
    prediction, msg_type, symptoms = predict_disease(image)

    if msg_type == "success":
        st.success(prediction)
    else:
        st.error(prediction)
        if symptoms:
            st.warning(f"**Symptoms:** \n{symptoms}")
