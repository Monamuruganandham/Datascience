import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Load the trained model
model_path = r"C:\Users\Mohan\PycharmProjects\pythonProject\miniproject 3\passenger_random_forest_model.pkl"
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None
#https://images.unsplash.com/photo-1498098662025-04e60a212db4?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8ZmxpZ2h0JTIwY3VzdG9tZXJzfGVufDB8fDB8fHww

if model is None:
    st.write("Model failed to load. Please check the model file.")
    st.stop()

# Feature list (expected by the model)
expected_features = [
    "Age",
    "Class",
    "Flight Distance",
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
    "Gender_Male",
    "Customer Type_disloyal Customer",
    "Type of Travel_Personal Travel",
]

# Streamlit app title
st.markdown(f"""
    <style>
    .stApp {{
            background-image: url("https://images.unsplash.com/photo-1437846972679-9e6e537be46e?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTF8fGZsaWdodCUyMGN1c3RvbWVyc3xlbnwwfHwwfHx8MA%3D%3D");
            
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            
    }}
    
    .stSidebar {{
        background-image: url("https://images.unsplash.com/photo-1437846972679-9e6e537be46e?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTF8fGZsaWdodCUyMGN1c3RvbWVyc3xlbnwwfHwwfHx8MA%3D%3D");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    .centered-title {{
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            color: #333;
    }}
    .gap {{
            margin-top: 40px; /* Adjust the gap size as needed */
    }}
    .centered-button {{
            display: flex;
            justify-content: center;
            margin-top: 20px;
    }}
    </style>
    """, unsafe_allow_html=True)
st.markdown('<div class="centered-title">CUSTOMER SATISFACTION PREDICTION APP</div>', unsafe_allow_html=True)
st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
st.sidebar.header("**INPUT FEATURES**")
st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
# Function to get user input
def user_input_features():
    # Numeric sliders
    age = st.sidebar.slider("**Age**", 0, 100, 30)
    flight_distance = st.sidebar.slider("**Flight Distance (in km)**", 0, 5000, 1000)
    departure_delay = st.sidebar.slider("**Departure Delay (in minutes)**", 0, 300, 0)
    arrival_delay = st.sidebar.slider("**Arrival Delay (in minutes)**", 0, 300, 0)

    # Service ratings (0-5)
    inflight_wifi = st.sidebar.selectbox("**Inflight Wifi Service**", [0, 1, 2, 3, 4, 5])
    time_convenient = st.sidebar.selectbox("**Departure/Arrival Time Convenient**", [0, 1, 2, 3, 4, 5])
    online_booking = st.sidebar.selectbox("**Ease of Online Booking**", [0, 1, 2, 3, 4, 5])
    gate_location = st.sidebar.selectbox("**Gate Location**", [0, 1, 2, 3, 4, 5])
    food_and_drink = st.sidebar.selectbox("**Food and Drink**", [0, 1, 2, 3, 4, 5])
    online_boarding = st.sidebar.selectbox("**Online Boarding**", [0, 1, 2, 3, 4, 5])
    seat_comfort = st.sidebar.selectbox("**Seat Comfort**", [0, 1, 2, 3, 4, 5])
    inflight_entertainment = st.sidebar.selectbox("**Inflight Entertainment**", [0, 1, 2, 3, 4, 5])
    onboard_service = st.sidebar.selectbox("**On-board Service**", [0, 1, 2, 3, 4, 5])
    legroom_service = st.sidebar.selectbox("**Leg Room Service**", [0, 1, 2, 3, 4, 5])
    baggage_handling = st.sidebar.selectbox("**Baggage Handling**", [0, 1, 2, 3, 4, 5])
    checkin_service = st.sidebar.selectbox("**Check-in Service**", [0, 1, 2, 3, 4, 5])
    inflight_service = st.sidebar.selectbox("**Inflight Service**", [0, 1, 2, 3, 4, 5])
    cleanliness = st.sidebar.selectbox("**Cleanliness**", [0, 1, 2, 3, 4, 5])

    # Categorical (binary encoded)
    gender = st.sidebar.selectbox("**Gender**", ["Male", "Female"])
    customer_type = st.sidebar.selectbox("**Customer Type**", ["Disloyal Customer", "Loyal Customer"])
    type_of_travel = st.sidebar.selectbox("**Type of Travel**", ["Personal Travel", "Business Travel"])
    travel_class = st.sidebar.selectbox("**Class**", ["Eco", "Eco Plus", "Business"])

    # One-hot encoding for categorical features
    data = {
        "Age": age,
        "Flight Distance": flight_distance,
        "Departure Delay in Minutes": departure_delay,
        "Arrival Delay in Minutes": arrival_delay,
        "Inflight wifi service": inflight_wifi,
        "Departure/Arrival time convenient": time_convenient,
        "Ease of Online booking": online_booking,
        "Gate location": gate_location,
        "Food and drink": food_and_drink,
        "Online boarding": online_boarding,
        "Seat comfort": seat_comfort,
        "Inflight entertainment": inflight_entertainment,
        "On-board service": onboard_service,
        "Leg room service": legroom_service,
        "Baggage handling": baggage_handling,
        "Checkin service": checkin_service,
        "Inflight service": inflight_service,
        "Cleanliness": cleanliness,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Customer Type_disloyal Customer": 1 if customer_type == "Disloyal Customer" else 0,
        "Type of Travel_Personal Travel": 1 if type_of_travel == "Personal Travel" else 0,
        "Class_Business": 1 if travel_class == "Business" else 0,
        "Class_Eco Plus": 1 if travel_class == "Eco Plus" else 0,
        "Class_Eco": 1 if travel_class == "Eco" else 0,
    }



    return pd.DataFrame(data, index=[0])

# Collect user input
input_df = user_input_features()

# Reindex input DataFrame to match expected features
input_df = input_df.reindex(columns=expected_features, fill_value=0)

# Display user input
st.subheader("User Input Features")
st.write(input_df)
st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

page = st.radio("**SELECT PAGE**", ["Customer Satisfaction Prediction", "Customer Satisfaction Trends"])
st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

if page == "Customer Satisfaction Prediction":
    # Display the "Predict Satisfaction" button only on this page
    with st.container():
        st.markdown('<div class="centered-button">', unsafe_allow_html=True)
        if st.button("Predict Satisfaction", key="predict_button"):
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            result = "Satisfied" if prediction[0] == 1 else "Neutral or Dissatisfied"
            st.subheader("Prediction Result")
            st.write(f"Customer Satisfaction Level: **{result}**")

            st.subheader("Prediction Probability")
            st.write(f"Satisfied: {prediction_proba[0][1]:.2f}, Neutral or Dissatisfied: {prediction_proba[0][0]:.2f}")

            st.subheader("CUSTOMER SATISFACTION DEMOGRAPHIC")
            categories = ["Satisfied", "Neutral or Dissatisfied"]
            probabilities = prediction_proba[0]

            # Pie chart
            fig, ax = plt.subplots()
            ax.pie(probabilities, labels=categories, autopct='%1.1f%%', startangle=90, colors=["pink", "blue"])
            ax.axis("equal")
            st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Customer Satisfaction Trends":
    # Function to display trends
    def display_trends():
        st.subheader("CUSTOMER SATISFACTION TRENDS")

        # Example dataset (replace with real dataset if available)
        sample_data = pd.DataFrame({
            "Category": ["Satisfied", "Neutral/Dissatisfied"],
            "Proportion": [0.65, 0.35],
        })

        # Example satisfaction by class
        class_data = pd.DataFrame({
            "Class": ["Eco", "Eco Plus", "Business"],
            "Satisfied": [70, 80, 95],
            "Neutral/Dissatisfied": [30, 20, 5],
        })

        st.write("**CUSTOMER SATISFACTION BY CLASS**")
        class_data.set_index("Class")[["Satisfied", "Neutral/Dissatisfied"]].plot(kind="bar", stacked=True, color=["violet", "yellow"], figsize=(10, 4))
        plt.title("Satisfaction Levels by Class")
        plt.ylabel("Percentage")
        plt.xlabel("Class")
        plt.legend(
            title="Satisfaction Level",
            bbox_to_anchor=(1, 1),  # Position at top-right
            loc='upper left'  # Align the legend at the top-left of the legend box
        )
        st.pyplot(plt)

        # Example satisfaction by type of travel
        travel_data = pd.DataFrame({
            "Type of Travel": ["Business Travel", "Personal Travel"],
            "Satisfied": [90, 60],
            "Neutral/Dissatisfied": [10, 40],
        })

        st.write("**CUSTOMER SATISFACTION BY TYPE OF TRAVEL**")
        travel_data.set_index("Type of Travel")[["Satisfied", "Neutral/Dissatisfied"]].plot(kind="bar", stacked=True, color=["violet", "yellow"], figsize=(10, 4))
        plt.title("Satisfaction Levels by Type of Travel")
        plt.ylabel("Percentage")
        plt.xlabel("Type of Travel")
        plt.legend(
            title="Satisfaction Level",
            bbox_to_anchor=(1, 1),  # Position at top-right
            loc='upper left'  # Align the legend at the top-left of the legend box
        )
        st.pyplot(plt)

    display_trends()