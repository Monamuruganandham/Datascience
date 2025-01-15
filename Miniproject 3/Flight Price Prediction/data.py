import numpy as np
import pandas as pd
import pickle
import joblib
import streamlit as st
from datetime import datetime

# Provide the full path to the model
model_path = r"C:\Users\Mohan\PycharmProjects\pythonProject\miniproject 3\final_random_forest_model.pkl"
try:
    model = joblib.load(open(model_path, 'rb'))
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None

# Check if the model is loaded correctly
if model is None:
    st.write("Model failed to load. Please check the model file.")


st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://images.unsplash.com/photo-1471922694854-ff1b63b20054?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8YmVhdXRpZnVsJTIwYmFja2dyb3VuZHxlbnwwfHwwfHx8MA%3D%3D");
        background-attachment: fixed;
        background-size: cover;
    }}
    .center-button {{
        display: flex;
        justify-content: flex-end;
        align-items: center;
        position: absolute;
        bottom: 30px;
        width: 100%;
        z-index: 999;
        margin-top: 100px;
        right: 20px;
    }}
    .gap {{
          height: 100px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
plane_image ="https://cdn-icons-png.flaticon.com/512/7893/7893979.png"
st.markdown(f'<h1><img src="{plane_image}" width="50" style="vertical-align:middle;">FLIGHT PRICE PREDICTION </h1>', unsafe_allow_html=True)


# Define mappings for categorical values
source_mapping = {'Banglore': 0, 'Kolkata': 1, 'Delhi': 2, 'Chennai': 3, 'Mumbai': 4}
selected_source = st.selectbox('**Select Source**', source_mapping)
source = source_mapping[selected_source]

destination_mapping = {'New Delhi': 0, 'Banglore': 1, 'Cochin': 2, 'Kolkata': 3, 'Delhi': 4, 'Hyderabad': 5}
selected_destination = st.selectbox('**Select Destination**', destination_mapping)
destination = destination_mapping[selected_destination]

airline_mapping = {
    'IndiGo': 0, 'Air India': 1, 'Jet Airways': 2, 'SpiceJet': 3,
    'Multiple carriers': 4, 'GoAir': 5, 'Vistara': 6, 'Air Asia': 7,
    'Vistara Premium economy': 8, 'Jet Airways Business': 9,
    'Multiple carriers Premium economy': 10, 'Trujet': 11
}
selected_airline = st.selectbox('**Select Airline**', airline_mapping)
airline = airline_mapping[selected_airline]

# Number of Stops
total_stops = st.slider('**Number of Stops**', min_value=0, max_value=4, step=1)

# Date and Time Inputs

dep_date = st.date_input("**Date of Journey**", value=datetime.now().date())
time_range = pd.date_range("00:20", "23:55", freq="5T").strftime('%H:%M')
dep_time = st.sidebar.selectbox("**Departure Time**", time_range)
arrival_date = dep_date
time_range1 = pd.date_range("00:05", "23:55", freq="5T").strftime('%H:%M')
arrival_time1 = st.sidebar.selectbox("**Arrival Time**", time_range1)


# Extracting features from the input data
Journey_Day = dep_date.day
Journey_Month = dep_date.month
Journey_Year = dep_date.year
dep_time_obj = datetime.strptime(dep_time, '%H:%M')
Departure_Hour = dep_time_obj.hour
Departure_Min = dep_time_obj.minute
arr_time_obj = datetime.strptime(arrival_time1, '%H:%M')
Arrival_Hour = arr_time_obj.hour
Arrival_Min = arr_time_obj.minute

additional_info_mapping = {
    'No info': 0, 'In-flight meal not included': 1, 'No check-in baggage included': 2,
    '1 Short layover': 3, 'No Info': 4, '1 Long layover': 5, 'Change airports': 6,
    'Business class': 7, 'Red-eye flight': 8, '2 Long layover': 9
}

# Select additional info

additional_info_encoded = 0
arrival_time = arr_time_obj.time()
depa_time=dep_time_obj.time()

# Calculate duration
Departure_Datetime = datetime.combine(dep_date, depa_time)
Arrival_Datetime = datetime.combine(arrival_date, arrival_time)
duration = Arrival_Datetime - Departure_Datetime

Duration_Hours = duration.days * 24 + duration.seconds // 3600
Duration_Minutes = (duration.seconds % 3600) // 60


st.markdown("<div class='gap'></div>", unsafe_allow_html=True)
# Ensure model expects the correct number of features
with st.markdown("<div class='center-button'>", unsafe_allow_html=True):
    if st.button("Predict Flight Price"):
        # Prepare data for prediction: 14 features
        features = [
            airline, source, destination, total_stops, additional_info_encoded,
            Journey_Day, Journey_Month, Journey_Year,
            Duration_Hours, Duration_Minutes,
            Departure_Hour, Departure_Min,
            Arrival_Hour, Arrival_Min
        ]

        # Convert features to numpy array and check the shape
        features = np.array(features).reshape(1, -1)
        if features.shape[1] != model.n_features_in_:
            st.error(f"Feature mismatch! Expected {model.n_features_in_} features, got {features.shape[1]}.")
        else:
            try:
                prediction = model.predict(features)
                predicted_price = prediction[0]

                # Set the color for the predicted price (you can change "blue" to any color)
                price_color = "brown"  # You can change this to "green", "red", or any color code

                # Display the price in the specified color
                st.markdown(f'<h3 style="color:{price_color};">PREDICTED FLIGHT PRICE: ₹{predicted_price:,.2f}</h3>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
st.markdown('</div>', unsafe_allow_html=True)
