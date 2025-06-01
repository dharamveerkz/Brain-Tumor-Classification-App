import streamlit as st
import os
import torch
from utils.model_utils import load_models, load_image

# Load models
model_scratch, model_pretrained, model_cbam = load_models()

# Define the class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Function to predict
def predict(model, image_tensor, class_names):
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]


# Streamlit app
def main():
    st.title("Brain Tumor Classification App")
    st.write("Welcome to the Brain Tumor Classification App. Please upload an MRI image for prediction.")

    # Upload image
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        image_tensor, image = load_image(file_path)

        # Predict and display results for each model
        st.subheader("Predictions from All Models")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("### ResNet-18 (Scratch)")
            prediction_scratch = predict(model_scratch, image_tensor)
            st.image(image, caption="Uploaded MRI Image", use_container_width=True)
            st.write(f"**Predicted Class:** {prediction_scratch}")

        with col2:
            st.write("### ResNet-18 (Pretrained)")
            prediction_pretrained = predict(model_pretrained, image_tensor)
            st.image(image, caption="Uploaded MRI Image", use_container_width=True)
            st.write(f"**Predicted Class:** {prediction_pretrained}")

        with col3:
            st.write("### ResNet-18 + CBAM")
            prediction_cbam = predict(model_cbam, image_tensor)
            st.image(image, caption="Uploaded MRI Image", use_container_width=True)
            st.write(f"**Predicted Class:** {prediction_cbam}")

if __name__ == "__main__":
    main()