#Create a Streamlit Web App
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the Iris species
species = ['Setosa', 'Versicolor', 'Virginica']

# Streamlit app title
st.title("Iris Flower Species Prediction")

# Create input fields for the four features
st.header("Enter the Iris Flower Measurements")
sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.0, step=0.1)
sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=5.0, value=3.0, step=0.1)
petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.0, step=0.1)
petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

# Create a button to make the prediction
if st.button("Predict"):
    # Prepare the input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_df = pd.DataFrame(input_data, columns=['sepal length (cm)', 'sepal width (cm)', 
                                                 'petal length (cm)', 'petal width (cm)'])

    # Make the prediction
    prediction = model.predict(input_df)
    predicted_species = species[prediction[0]]

    # Display the result
    st.success(f"The predicted Iris species is: **{predicted_species}**")