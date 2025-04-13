from tensorflow.keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st

# Function to classify the fruit
def classify_images(img):
    np.set_printoptions(suppress=True)  # Disable scientific notation

    # Load the model
    model = load_model("model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create input array for the N
    data = np.ndarray(shape=(1, 128, 128, 3), dtype=np.float32)

    # Convert image to RGB and resize
    #image = img.convert("RGB")
    size = (128, 128)
    image = ImageOps.fit(img, size)
    #image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert image to numpy array and normalize
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 255.0)
    #normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name.strip(), confidence_score

# Streamlit App Configuration
st.set_page_config(layout="wide")

st.title("DEEP FAKE IMAGE DETECTOR")

# Sidebar: Display sample fruits
st.sidebar.write("# Image Sample")
#st.sidebar.write("Drag and drop images from below for classification.")

st.sidebar.write("### Deep Fake Images")
cols = st.sidebar.columns(2)  # Create 2 columns for images in a row

# DeepFake Images
spoiled_images = ["images/fake1.JPG", "images/fake2.jpg", "images/fake3.jpg", "images/fake4.jpg"]
spoiled_captions = ["Fake", "Fake", "Fake", "Fake"]

for idx, img_path in enumerate(spoiled_images):
    with cols[idx % 2]:  # Cycle through columns
        st.image(img_path, caption=spoiled_captions[idx], use_container_width=True)


# Use columns in the sidebar to align images with spacing
st.sidebar.write("### Real Images")
cols = st.sidebar.columns(2)  # Create 2 columns for images in a row

# Real Images
fresh_images = ["images/real1.JPG", "images/real2.jpg", "images/real3.JPG", "images/real4.jpg"]
fresh_captions = ["Real", "Real", "Real", "Real"]

for idx, img_path in enumerate(fresh_images):
    with cols[idx % 2]:  # Cycle through columns
        st.image(img_path, caption=fresh_captions[idx], use_container_width=True)




# Image Upload
input_img = st.file_uploader("Upload or Drag & Drop an image", type=["jpg", "png", "jpeg"])

if input_img is not None:
    if st.button("Classify"):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.info("Your Uploaded Image")
            st.image(input_img, use_container_width=False, width=200)  # Smaller image

        with col2:
            st.info("Classification Result")
            image_file = Image.open(input_img)
            label, confidence_score = classify_images(image_file)
            print(confidence_score)

            if label.startswith("1"):
                st.info("Result: Real Image")

                # Barre de progression pour "Real Face"
                st.markdown(f"Real {int(confidence_score * 100)}%")
                st.progress(int(confidence_score * 100))

                # Affichage du pourcentage
                #st.write(f"{int(confidence_score * 100)}%")

                # Barre de progression pour "Face Manipulated"
                st.markdown(f"Fake {(100 - int(confidence_score * 100))}%")
                st.progress(100 - int(confidence_score * 100))

                # Affichage du pourcentage
                #st.write(f"{(100 - int(confidence_score * 100))}%")
            elif label.startswith("0"):
                st.error("Result: DeepFake Image")

                # Barre de progression pour "Real Face"
                st.markdown(f"Real {(100 - int(confidence_score * 100))}%")
                st.progress(100 - int(confidence_score * 100))

                # Affichage du pourcentage
                #st.write(f"{int(confidence_score * 100)}%")

                # Barre de progression pour "Face Manipulated"
                st.markdown(f"Fake {int(confidence_score * 100)}%")
                st.progress(int(confidence_score * 100))

                # Affichage du pourcentage
                #st.write(f"{(100 - int(confidence_score * 100))}%")
            else:
                st.error("The image could not be classified into any relevant category.")
            
            