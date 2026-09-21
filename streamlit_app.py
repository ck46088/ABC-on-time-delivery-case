
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('delivery_delay.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Define the input features based on x.columns
# x.columns: Index(['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
#        'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
#        'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
#        'Warehouse_Processing_Time'], dtype='object')

delivery_distance = st.slider('Delivery Distance (km)', 0.0, 100.0, 25.0)
traffic_congestion = st.selectbox('Traffic Congestion (1=Low, 5=High)', [1, 2, 3, 4, 5], index=2)
weather_condition = st.selectbox('Weather Condition (1=Good, 5=Bad)', [1, 2, 3, 4, 5], index=0)
delivery_slot = st.selectbox('Delivery Slot (1=Morning, 2=Afternoon, 3=Evening)', [1, 2, 3], index=1)
driver_experience = st.slider('Driver Experience (Years)', 0, 30, 5)
num_stops = st.slider('Number of Stops', 1, 20, 5)
vehicle_age = st.slider('Vehicle Age (Years)', 0, 15, 3)
road_condition_score = st.selectbox('Road Condition Score (1=Poor, 5=Excellent)', [1, 2, 3, 4, 5], index=2)
package_weight = st.slider('Package Weight (kg)', 0.0, 50.0, 5.0)
fuel_efficiency = st.slider('Fuel Efficiency (km/l)', 5.0, 30.0, 15.0)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', 0, 120, 60)

# Create a DataFrame for prediction
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0]

    if prediction[0] == 1:
        st.error(f"Prediction: Delivery **Likely Delayed** (Probability: {prediction_proba[1]:.2f})")
    else:
        st.success(f"Prediction: Delivery **No Delay** (Probability: {prediction_proba[0]:.2f})")

    st.write("--- Recommended values to test --- ")
    st.write("No Delay: Delivery_Distance=10, Traffic_Congestion=1, Weather_Condition=1, Delivery_Slot=1, Driver_Experience=10, Num_Stops=2, Vehicle_Age=2, Road_Condition_Score=5, Package_Weight=5, Fuel_Efficiency=20, Warehouse_Processing_Time=30")
    st.write("Delayed: Delivery_Distance=50, Traffic_Congestion=5, Weather_Condition=5, Delivery_Slot=3, Driver_Experience=2, Num_Stops=15, Vehicle_Age=10, Road_Condition_Score=1, Package_Weight=40, Fuel_Efficiency=8, Warehouse_Processing_Time=100")
