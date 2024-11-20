#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
from keras import models, layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from keras.models import load_model
#from sklearn.preprocessing import LogisticRegression()
from sklearn.linear_model import LogisticRegression

# Define the model save path
MODEL_PATH = "heart_disease_model.h5"

# Function to save the trained model
def save_model(model):
    try:
        model.save(MODEL_PATH)
        st.success("Model saved successfully!")
    except Exception as e:
        st.error(f"Error while saving the model: {e}")

# Function to load the saved model


# Function to load the saved model
def load_trained_model():
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            # Recompile the model after loading to avoid the warning
            model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
            st.success("Model loaded and recompiled successfully!")
            return model
        else:
            st.warning("No saved model found. Training a new model.")
            return None
    except Exception as e:
        st.error(f"Error while loading the model: {e}")
        return None

# Build the model
# def build_model(input_shape):
#     try:
#         model = models.Sequential()
#         model.add(layers.Input(shape=(input_shape,)))
#         model.add(layers.Dense(16, activation='relu'))
#         model.add(layers.Dense(16, activation='relu'))
#         model.add(layers.Dense(16, activation='relu'))
#         model.add(layers.Dense(1, activation='sigmoid'))
#         model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#         return model
#     except Exception as e:
#         st.error(f"Error while building the model: {e}")
#         return None
# Function to build the primary neural network model and a logistic regression model for showcase
def build_model(input_shape):
    try:
        
        model = models.Sequential()
        model.add(layers.Input(shape=(input_shape,)))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Logistic regression model (showcase only, not used in the app)
        logistic_model = models.Sequential()
        logistic_model.add(layers.Input(shape=(input_shape,)))
        logistic_model.add(layers.Dense(1, activation='sigmoid'))
        logistic_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])


        return model  # Returning only the primary model (neural network)
    except Exception as e:
        st.error(f"Error while building the model: {e}")
        return None

# Load and prepare the data
@st.cache_data
def load_data():
    data = pd.read_csv('heart.csv')
    return data

# Sidebar for navigation
st.sidebar.header("Navigation")
menu_options = ["Home", "Going Through Data", "Predict for a Patient"]
selected_option = st.sidebar.selectbox("Choose an option", menu_options)

# Load data
data = load_data()

# Split data into features and target
Features = data.iloc[:, :-1]
Target = data.iloc[:, -1]

# Splitting data into training and test sets
train_Features, test_Features, train_target, test_target = train_test_split(Features, Target, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_Features)
X_test = scaler.transform(test_Features)

# Ensure the target is in the right shape
train_target = train_target.values.ravel()
test_target = test_target.values.ravel()


# Load the pre-trained model or train a new one
model = load_trained_model()

if model is None:
    # Build and train the model if it's not loaded
    model = build_model(train_Features.shape[1])
    if model:
        model.fit(X_train, train_target, epochs=50, batch_size=10, validation_split=0.2)
        # Save the model after training
        save_model(model)

# Home page
if selected_option == "Home":
    st.title("🏠 Heart Disease Prediction App")
    
    # Introduction to the app
    st.header("Introduction")
    st.write("""Welcome to the **Heart Disease Prediction App**!""")
    
    # How it works
    st.subheader("How it Works:")
    st.write("""The app uses a ML model trained on the **Heart Disease dataset**.""")

    #
# Going Through Data
elif selected_option == "Going Through Data":
    st.title("📊 Going Through Data")
    
    # Display Dataset Shape
    st.write("### Dataset Shape:", data.shape)
    
    # Display the dataset description
    st.write(data.describe())
    
    # Correlation Plot using Matplotlib
    st.subheader("Correlation Matrix")
    correlation_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(correlation_matrix, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=90)
    ax.set_yticklabels(correlation_matrix.columns)
    st.pyplot(fig)

# Prediction for a Patient
# elif selected_option == "Predict for a Patient":
#     st.title("🔧 Making Prediction")
#     st.subheader("Predict Heart Disease for a New Patient")
    
#     input_data = {
#         'age': st.number_input("Enter Age", min_value=0, max_value=100, value=50),
#         'sex': st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1]),
#         'cp': st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0),
#         'trestbps': st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120),
#         'chol': st.number_input("Cholesterol Level", min_value=0, max_value=600, value=200),
#         'fbs': st.selectbox("Fasting Blood Sugar (1 = >120mg/dl, 0 = otherwise)", [0, 1]),
#         'restecg': st.number_input("Resting ECG Result (0-2)", min_value=0, max_value=2, value=1),
#         'thalach': st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150),
#         'exang': st.selectbox("Exercise-induced Angina (1 = Yes, 0 = No)", [0, 1]),
#         'oldpeak': st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0),
#         'slope': st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=1),
#         'ca': st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0),
#         'thal': st.number_input("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", min_value=1, max_value=3, value=1),
#     }

#     input_df = pd.DataFrame([input_data])
#     input_scaled = scaler.transform(input_df)

#     if st.button("Predict"):
#         predicted_probability = model.predict(input_scaled)[0][0]
#         prediction = "Yes! Patient is predicted to be suffering from heart disease." if predicted_probability > 0.5 else "No! Patient is predicted not to be suffering from heart disease."
#         st.write(f"### Prediction Result: {prediction}")
#         st.write(f"Predicted Probability of Heart Disease: {predicted_probability:.2f}")
#         st.subheader("Prediction Probability")
#         st.progress(float(predicted_probability))
#         st.write(predicted_probability)
# Prediction for a Patient
elif selected_option == "Predict for a Patient":
    st.title("🔧 Making Prediction")
    st.subheader("Predict Heart Disease for a New Patient")
    
    # Collecting user input
    input_data = {
        'age': st.number_input("Enter Age", min_value=0, max_value=100, value=50),
        'sex': st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1]),
        'cp': st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0),
        'trestbps': st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120),
        'chol': st.number_input("Cholesterol Level", min_value=0, max_value=600, value=200),
        'fbs': st.selectbox("Fasting Blood Sugar (1 = >120mg/dl, 0 = otherwise)", [0, 1]),
        'restecg': st.number_input("Resting ECG Result (0-2)", min_value=0, max_value=2, value=1),
        'thalach': st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150),
        'exang': st.selectbox("Exercise-induced Angina (1 = Yes, 0 = No)", [0, 1]),
        'oldpeak': st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0),
        'slope': st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=1),
        'ca': st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0),
        'thal': st.number_input("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", min_value=1, max_value=3, value=1),
    }

    # Preparing data for prediction
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    if st.button("Predict"):
        # Predict the probability
        predicted_probability = model.predict(input_scaled)[0][0]
        prediction = (
            "✅ **No!** The patient is predicted **not** to have heart disease."
            if predicted_probability <= 0.5
            else "⚠️ **Yes!** The patient is predicted to have heart disease."
        )
        
        # Display the result with a styled UI
        st.subheader("Prediction Result")
        st.markdown(
            f"<div style='border: 1px solid #ccc; padding: 15px; border-radius: 10px; "
            f"background-color: {'#d4edda' if predicted_probability <= 0.5 else '#f8d7da'};'>"
            f"<strong>{prediction}</strong></div>",
            unsafe_allow_html=True,
        )

        # Display the probability with styling
        st.subheader("Prediction Probability")
        st.progress(float(predicted_probability))
        st.markdown(
            f"<h3 style='color: #4CAF50;'>{predicted_probability:.2%}</h3>"
            if predicted_probability <= 0.5
            else f"<h3 style='color: #E53935;'>{predicted_probability:.2%}</h3>",
            unsafe_allow_html=True,
        )
