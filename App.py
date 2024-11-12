import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('MIPML.pkl', 'rb') as file:
    model = pickle.load(file)

# Page Title and Description
st.title("🌸 Iris Flower Species Prediction App")
st.write("""
This app predicts the species of an iris flower based on its **sepal length**, **sepal width**, **petal length**, and **petal width**.
Simply adjust the sliders to input the flower's measurements, and click **Predict** to see the result!
""")

# Sidebar for user guidance
st.sidebar.header("Guide to Input Features")
st.sidebar.write("""
- **Sepal Length**: The length of the sepal in centimeters.
- **Sepal Width**: The width of the sepal in centimeters.
- **Petal Length**: The length of the petal in centimeters.
- **Petal Width**: The width of the petal in centimeters.
- Adjust the values to match the flower's characteristics.
""")

# Input fields for features
st.header("Enter Flower Measurements")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, step=0.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, step=0.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, step=0.1)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, step=0.1)

# Adding details about species based on feature ranges
st.header("Hints for Each Species")
st.write("""
- **Iris-setosa**: Generally has smaller petal lengths and widths. Often, petal length is below 2 cm.
- **Iris-versicolor**: Intermediate-sized petals and sepals, often with petal lengths between 3 to 5 cm.
- **Iris-virginica**: Typically has the largest petals, with lengths often over 5 cm and widths over 1.5 cm.
""")

# Prediction button and result display
if st.button("Predict"):
    # Prepare the input data as a 2D array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Perform the prediction
    prediction = model.predict(input_data)
    
    # Mapping the output to the correct species name
    species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    predicted_species = species[prediction[0]]
    
    # Displaying the result
    st.success(f"The predicted species is: **{predicted_species}**")
    
    # Additional details based on prediction
    if predicted_species == "Iris-setosa":
        st.write("**Iris-setosa** is known for its small petal length and width.")
    elif predicted_species == "Iris-versicolor":
        st.write("**Iris-versicolor** has intermediate petal measurements compared to the other two species.")
    else:
        st.write("**Iris-virginica** has larger petal lengths and widths, making it the largest species in this dataset.")
