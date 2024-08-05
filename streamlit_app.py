import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv('solarpowergeneration.csv')
    df.dropna(inplace=True)  # Handle missing values
    return df

df = load_data()

# Display the dataset
st.title("Solar Power Generation Data")
st.write("### Dataset Preview")
st.dataframe(df.head())

# Display summary statistics
st.write("### Summary Statistics")
st.write(df.describe())

# Feature selection and target variable
X = df[['distance-to-solar-noon', 'temperature', 'wind-direction', 'wind-speed', 'sky-cover', 
        'visibility', 'humidity', 'average-wind-speed-(period)', 'average-pressure-(period)']]
y = df['power-generated']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
@st.cache
def train_model():
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

model = train_model()

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display model performance
st.write("### Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-Squared (R2): {r2:.2f}")

# Save the model
joblib.dump(model, 'solar_power_model.pkl')

# Prediction section
st.write("### Make Predictions")

# Input fields for prediction
distance_to_solar_noon = st.number_input("Distance to Solar Noon", value=50.0)
temperature = st.number_input("Temperature", value=25)
wind_direction = st.number_input("Wind Direction", value=180)
wind_speed = st.number_input("Wind Speed", value=5.0)
sky_cover = st.number_input("Sky Cover", value=0)
visibility = st.number_input("Visibility", value=10.0)
humidity = st.number_input("Humidity", value=40)
average_wind_speed = st.number_input("Average Wind Speed", value=3.0)
average_pressure = st.number_input("Average Pressure", value=1015.0)

# Prediction button
if st.button("Predict Power Generated"):
    input_data = pd.DataFrame({
        'distance-to-solar-noon': [distance_to_solar_noon],
        'temperature': [temperature],
        'wind-direction': [wind_direction],
        'wind-speed': [wind_speed],
        'sky-cover': [sky_cover],
        'visibility': [visibility],
        'humidity': [humidity],
        'average-wind-speed-(period)': [average_wind_speed],
        'average-pressure-(period)': [average_pressure]
    })
    prediction = model.predict(input_data)[0]
    st.write(f"Predicted Power Generated: {prediction:.2f} units")

# Streamlit app settings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Solar Power Generation Prediction App')
st.write("Use the inputs to predict the power generated.")
