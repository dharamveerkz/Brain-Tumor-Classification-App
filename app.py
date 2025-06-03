import streamlit as st
import os
import torch
from PIL import Image
from utils.model_utils import load_models, load_image

# Load models
model_scratch, model_pretrained, model_cbam = load_models()

# Define the class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Function to predict class
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]

# Streamlit app
def main():
    st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")
    st.title("Brain Tumor Classification App")
    st.write("Upload an MRI scan image (.jpg, .jpeg, or .png) to get predictions from three deep learning models.")

    # Upload image
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save uploaded file to temp directory
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Load image and preprocess
            image_tensor, image = load_image(file_path)

            # Display uploaded image
            st.image(image, caption="Uploaded MRI Image", use_container_width=True)

            st.markdown("---")
            st.subheader("Predictions from All Models")

            # Display model results in columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### ResNet-18 (Scratch)")
                prediction = predict(model_scratch, image_tensor)
                st.success(f"Predicted: {prediction.upper()}")

            with col2:
                st.markdown("### ResNet-18 (Pretrained)")
                prediction = predict(model_pretrained, image_tensor)
                st.success(f"Predicted: {prediction.upper()}")

            with col3:
                st.markdown("### ResNet-18 + CBAM")
                prediction = predict(model_cbam, image_tensor)
                st.success(f"Predicted: {prediction.upper()}")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main()
