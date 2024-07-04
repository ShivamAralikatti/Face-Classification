import streamlit as st
import face_recognition
import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load the known faces and encodings
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

def recognize_celebrities(image, data):
    image_np = np.array(image)
    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)

    names = []
    similarities = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        similarity = 0

        face_distances = face_recognition.face_distance(data["encodings"], encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = data["names"][best_match_index]
            similarity = (1 - face_distances[best_match_index]) * 100  # Convert distance to similarity percentage

        names.append(name)
        similarities.append(similarity)

    return face_locations, names, similarities

def display_image_with_names(image, face_locations, names, similarities, threshold=70):
    image_np = np.array(image)
    fig, ax = plt.subplots()
    ax.imshow(image_np)

    found = False
    for (top, right, bottom, left), name, similarity in zip(face_locations, names, similarities):
        if similarity >= threshold:
            rect = plt.Rectangle((left, top), right - left, bottom - top, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            plt.text(left, top - 10, f"{name}", color='green', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
            found = True

    plt.axis('off')
    return fig, found

# Custom HTML/CSS
custom_css = """
<style>
    .reportview-container {
        background: #F0F0F5;
    }
    .sidebar .sidebar-content {
        background: #E8E8EF;
    }
    .stButton>button {
        color: white;
        background-color: #1E90FF;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Streamlit interface
st.title("Celebrity Recognition 🎥")
st.markdown("Upload an image and find out the name of the famous celebrity!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    face_locations, names, similarities = recognize_celebrities(image, data)
    fig, found = display_image_with_names(image, face_locations, names, similarities, threshold=70)

    if found:
        st.pyplot(fig)
        st.write(f"**Recognized:** {names[0]} ({similarities[0]:.2f}% similarity)")
    else:
        st.markdown("### Sorry, this person is not in the database.")
