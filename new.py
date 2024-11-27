import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# MySQL connection string
db_connection_str = 'mysql+pymysql://root:Mona%40999@localhost/redbusproject'

# Create SQLAlchemy engine
engine = create_engine(db_connection_str)

# Read data from MySQL using pandas
query = "SELECT * FROM bus_details"
dfbus = pd.read_sql(query, engine)

# Convert Departing_time to datetime for easier filtering
dfbus['Departing_time'] = pd.to_datetime(dfbus['Departing_time'], format='%H:%M').dt.time

# Streamlit title and instructions
# Bus image URL
bus_image_url = "https://cdn-icons-png.flaticon.com/512/5030/5030991.png"

# Display the bus image along with the title using Markdown
st.markdown(f'<h1><img src="{bus_image_url}" width="50" style="vertical-align:middle;"> Redbus Data Scraping with Selenium & Dynamic Filtering using Streamlit</h1>', unsafe_allow_html=True)
st.write('This app allows you to filter and view bus route details. You can also download the filtered data.')


# Filter options
route_filter = st.multiselect('Select Route:', options=dfbus['Route_name'].unique())
#bus_type_filter = st.sidebar.multiselect(
    #"Bus Type",
    #["AC", "Non-AC", "Sleeper", "Semi-Sleeper", "Volvo"],  # Customize this list as per your data
    #default=[]
#)
bus_type_filter = st.sidebar.multiselect('Select Bus Type:', options=dfbus['Bus_type'].unique())
#bus_type_filter = st.multiselect('Select Bus Type (AC or Non-AC):', options=['AC', 'Non-AC'])
price_filter = st.sidebar.slider('Select Price Range:', min_value=int(dfbus['Price'].min()), max_value=int(dfbus['Price'].max()), value=(int(dfbus['Price'].min()), int(dfbus['Price'].max())))
rating_filter = st.sidebar.slider('Select Star Rating Range:', min_value=float(dfbus['Star_Rating'].min()), max_value=float(dfbus['Star_Rating'].max()), value=(float(dfbus['Star_Rating'].min()), float(dfbus['Star_Rating'].max())))

# Departing time filter
time_ranges = {
    '6 AM to 10 AM': (6, 10),
    '10 AM to 12 PM': (10, 12),
    '12 PM to 3 PM': (12, 15),
    '3 PM to 6 PM': (15, 18),
    '6 PM to 9 PM': (18, 21),
    '9 PM to 12 AM': (21, 24)
}

selected_time_range = st.selectbox('Select Departing Time Range:', list(time_ranges.keys()))

# Apply filters
filtered_data = dfbus

if route_filter:
    filtered_data = filtered_data[filtered_data['Route_name'].isin(route_filter)]

if bus_type_filter:
    filtered_data = filtered_data[filtered_data['Bus_type'].isin(bus_type_filter)]

filtered_data = filtered_data[(filtered_data['Price'] >= price_filter[0]) & (filtered_data['Price'] <= price_filter[1])]
filtered_data = filtered_data[(filtered_data['Star_Rating'] >= rating_filter[0]) & (filtered_data['Star_Rating'] <= rating_filter[1])]

# Filter by selected departing time range
start_hour, end_hour = time_ranges[selected_time_range]
filtered_data = filtered_data[filtered_data['Departing_time'].apply(lambda x: start_hour <= x.hour < end_hour)]

# Display filtered data
st.write('Filtered Bus Data:')
st.dataframe(filtered_data)

# Download button for filtered data as CSV
if not filtered_data.empty:
    st.download_button(
        label="Download Filtered Data",
        data=filtered_data.to_csv(index=False),
        file_name="filtered_bus_details.csv",
        mime="text/csv"
    )
else:
    st.warning("No data available with the selected filters.")

