import streamlit as st
import joblib
import pandas as pd
import altair as alt

# Load the model
model_path = r"C:\Users\Mohan\PycharmProjects\pythonProject\miniproject 3\passenger_random_forest_model.pkl"
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None

if model is None:
    st.write("Model failed to load. Please check the model file.")
    st.stop()



# Expected features in the model
expected_features = [
    "Gender",
    "Customer Type",
    "Age",
    "Type of Travel",
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
    "Arrival Delay in Minutes"
]

st.markdown(
    """
    <style>
    body {
        background-color: violet !important;
    }
    .stApp {
        background-color: violet !important;
    }
    .block-container {
        background-color: violet !important;
    }
    /* Change background color for sidebar */
    section[data-testid="stSidebar"] {
        background-color: violet !important;
    }
    div.stButton > button {
        display: block;
        margin: auto;
        background-color: darkblue;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 10px;
        border: none;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Streamlit app title
st.markdown(
    "<h1 style='text-align: center; color: darkblue;'>PASSENGER SATISFACTION PREDICTION</h1>",
    unsafe_allow_html=True
)



st.sidebar.header("RATINGS")

# Function to get user input and encode categorical features
def star_rating(label):
    """Function to display star ratings and return numerical values."""
    stars = st.sidebar.radio(label, ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], index=2)
    star_mapping = {"⭐": 1, "⭐⭐": 2, "⭐⭐⭐": 3, "⭐⭐⭐⭐": 4, "⭐⭐⭐⭐⭐": 5}
    return star_mapping[stars]

def user_input_features():
    # Numeric inputs
    age = st.slider("**Age**", 0, 100, 30)
    flight_distance = st.sidebar.slider("Flight Distance (in km)", 0, 5000, 1000)
    departure_delay = st.sidebar.slider("Departure Delay (min)", 0, 300, 0)
    arrival_delay = st.sidebar.slider("Arrival Delay (min)", 0, 300, 0)

    # Star-based ratings
    inflight_wifi = star_rating("Inflight Wifi Service")
    time_convenient = star_rating("Departure/Arrival Time Convenient")
    online_booking = star_rating("Ease of Online Booking")
    gate_location = star_rating("Gate Location")
    food_and_drink = star_rating("Food and Drink")
    online_boarding = star_rating("Online Boarding")
    seat_comfort = star_rating("Seat Comfort")
    inflight_entertainment = star_rating("Inflight Entertainment")
    onboard_service = star_rating("On-board Service")
    legroom_service = star_rating("Leg Room Service")
    baggage_handling = star_rating("Baggage Handling")
    checkin_service = star_rating("Check-in Service")
    inflight_service = star_rating("Inflight Service")
    cleanliness = star_rating("Cleanliness")

    # Categorical inputs
    gender = st.selectbox("**Gender**", ["Male", "Female"])
    customer_type = st.selectbox("**Customer Type**", ["Disloyal Customer", "Loyal Customer"])
    type_of_travel = st.selectbox("**Type of Travel**", ["Personal Travel", "Business Travel"])
    travel_class = st.selectbox("**Class**", ["Eco", "Eco Plus", "Business"])
    st.markdown("<br>", unsafe_allow_html=True)
    # Create DataFrame
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
        "Gender": gender,
        "Customer Type": customer_type,
        "Type of Travel": type_of_travel,
        "Class": travel_class
    }

    df = pd.DataFrame(data, index=[0])

    # One-hot encoding for categorical features
    df_encoded = pd.get_dummies(df, columns=["Gender", "Customer Type", "Type of Travel", "Class"])

    # Ensure the order of columns matches the expected features
    df_encoded = df_encoded.reindex(columns=expected_features, fill_value=0)

    return df_encoded

# Collect user input
input_df = user_input_features()
st.markdown("<br>", unsafe_allow_html=True)

# Prediction button
if st.button("**PREDICT SATISFACTION**"):
    if model:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        result = "Satisfied" if prediction[0] == 1 else "Neutral or Dissatisfied"
        st.subheader("PREDICTION RESULT")
        st.write(f"Customer Satisfaction Level: **{result}**")

        st.subheader("PREDICTION PROBABILITY")
        st.write(f"**Satisfied**: {prediction_proba[0][1]:.2f}")
        st.write(f"**Neutral or Dissatisfied**: {prediction_proba[0][0]: .2f}")

        # Update bar chart based on prediction
        satisfaction_trends = pd.DataFrame({
            "Category": ["Satisfied", "Neutral/Dissatisfied"],
            "Percentage": [prediction_proba[0][1] * 100, prediction_proba[0][0] * 100],
        })

        chart = (
            alt.Chart(satisfaction_trends)
            .mark_bar()
            .encode(
                x=alt.X("Category", sort=None, title="Satisfaction Level"),
                y=alt.Y("Percentage", title="Probability (%)"),
                color=alt.Color("Category",
                                scale=alt.Scale(domain=["Satisfied", "Neutral/Dissatisfied"], range=["blue", "pink"])),
                tooltip=["Category", "Percentage"]
            )
            .properties(width=500, height=300)
        )

        st.subheader("CUSTOMER SATISFACTION TRENDS")
        st.altair_chart(chart, use_container_width=True)

    else:
        st.error("Model not loaded properly. Check the model file.")


