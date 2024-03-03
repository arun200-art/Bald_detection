import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predict_baldness(img, model):
    img_array = image.img_to_array(img) / 255.0    # Convert to array and scale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    
    if prediction[0] > 0.5:
        return "Not Bald"
    else:
        return "Bald"

def main():
    st.title("Baldness Detection")

    # Load the model
    model = load_model('baldness_detection_model.h5')

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write("")

        prediction = predict_baldness(img, model)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
