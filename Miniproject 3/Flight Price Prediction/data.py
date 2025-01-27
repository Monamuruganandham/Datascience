import numpy as np
import pandas as pd
import pickle
import joblib
import streamlit as st
from datetime import datetime, timedelta

# Provide the full path to the model
model_path = r"C:\Users\Mohan\PycharmProjects\pythonProject\miniproject 3\final_gradient_model.pkl"
try:
    model = joblib.load(open(model_path, 'rb'))
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None

# Check if the model is loaded correctly
if model is None:
    st.write("Model failed to load. Please check the model file.")


def flight_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = model.predict(input_data_reshaped)

    rounded_value = round(prediction[0], 2)
    return rounded_value
def main():
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
    plane_image = "https://cdn-icons-png.flaticon.com/512/7893/7893979.png"
    st.markdown(f'<h1><img src="{plane_image}" width="50" style="vertical-align:middle;">FLIGHT PRICE PREDICTION </h1>',
                unsafe_allow_html=True)

    # Source
    sources = ['Chennai', 'Delhi', 'Kolkata', 'Mumbai']

    selected_source = st.selectbox('Select Source', sources)
    source_mapping = {source: 1 if source == selected_source else 0 for source in sources}
    Source = list(source_mapping.values())

    # Destination
    destinations = ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata']

    selected_destination = st.selectbox('Select Destination', destinations)
    destination_mapping = {destination: 1 if destination == selected_destination else 0 for destination in destinations}
    Destination = list(destination_mapping.values())

    dep_time = datetime.now().time()
    arrival_time = datetime.now().time()

    Journey_Day = datetime.now().day
    Journey_Month = datetime.now().month
    Departure_Hour = dep_time.hour
    Departure_Min = dep_time.minute
    Arrival_Hour = arrival_time.hour
    Arrival_Min = arrival_time.minute


    Duration_Hours = st.number_input("Duration Hours", min_value=0, max_value=47,step=1, value=2)
    Duration_Minutes = st.number_input("Duration Minutes", min_value=0, max_value=59, step=5, value=0)

    # Stops
    Total_Stops = st.number_input("Number of Stops", min_value=0, max_value=4,step=1, value=0)

    # Airline
    airlines = ['Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business', 'Multiple carriers',
                'Multiple carriers Premium economy', 'SpiceJet', 'Trujet', 'Vistara', 'Vistara Premium economy']

    selected_airline = st.selectbox('Select Airline', airlines)
    airline_mapping = {airline: 1 if airline == selected_airline else 0 for airline in airlines}
    Airlines = list(airline_mapping.values())

    journey_input = [Total_Stops, Journey_Day, Journey_Month, Departure_Hour, Departure_Min,
                     Arrival_Hour, Arrival_Min, Duration_Hours, Duration_Minutes]
    airline_input = Airlines
    source_input = Source
    destination_input = Destination

    Input = journey_input + airline_input + source_input + destination_input

    price_prediction = ''

    if st.button('Predict Price'):
        price_prediction = flight_prediction(Input)
        st.markdown(
            """
            <h3 style='text-align: center;'>You will have to pay approximately Rs. {}</h3>
            """.format(price_prediction),
            unsafe_allow_html=True
        )


if __name__ == '__main__':
    main()