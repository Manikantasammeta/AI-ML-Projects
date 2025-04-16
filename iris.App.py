import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd


# Load trained model
with open("my_model.pkl", "rb") as file:
    model = pickle.load(file)

# Flower images
flower_images = {
    'setosa': 'https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa_2.jpg',
    'versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
    'virginica': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
}

# Initialize session state to control visibility
if 'predicted' not in st.session_state:
    st.session_state.predicted = False

# Title
st.title("🌼 Iris Class Predictor")

# If not predicted, show sliders
if not st.session_state.predicted:
    st.subheader(" Enter flower measurements:")

    sepal_length = st.slider("Sepal Length (cm)", min_value=0.0, max_value=8.0, step=0.1)
    sepal_width = st.slider("Sepal Width (cm)", min_value=0.0, max_value=8.0, step=0.1)
    petal_length = st.slider("Petal Length (cm)", min_value=0.0, max_value=8.0, step=0.1)
    petal_width = st.slider("Petal Width (cm)", min_value=0.0, max_value=8.0, step=0.1)

    if st.button("Predict"):
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

        input_data = pd.DataFrame(scaler.transform(input_data))
        predicted_class_index = model.predict(input_data)
        print("Raw predicted index:", predicted_class_index)

        class_names = ['setosa', 'versicolor', 'virginica']
        predicted_class_name = class_names[predicted_class_index[0]]


        # Store results in session_state
        st.session_state.predicted = True
        st.session_state.inputs = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
        st.session_state.predicted_class = predicted_class_name

        st.rerun()  # Refresh the page to show result

# If predicted, show results
else:
    st.subheader(" Your Input:")
    st.write(f"- Sepal Length: {st.session_state.inputs['sepal_length']} cm")
    st.write(f"- Sepal Width: {st.session_state.inputs['sepal_width']} cm")
    st.write(f"- Petal Length: {st.session_state.inputs['petal_length']} cm")
    st.write(f"- Petal Width: {st.session_state.inputs['petal_width']} cm")

    st.subheader("🌸 Predicted Flower Class:")
    st.success(st.session_state.predicted_class)


    st.image(
        flower_images[st.session_state.predicted_class],
        caption=f"Iris {st.session_state.predicted_class}",
        width=300
    )

    if st.button("🔁 Predict Again"):
        st.session_state.predicted = False
        st.rerun()
