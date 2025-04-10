import streamlit as st
import tensorflow as tf
from PIL import Image
import os
import numpy as np
import google.generativeai as genai
import gdown

st.set_page_config(page_title="Smart Plant Disease Detection", layout="wide", page_icon="🌿")

st.markdown("""
<style>
    .main {
        background-color: #1e1e1e;
        color: #e0e0e0;
        padding: 2rem;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border-color: #4CAF50;
    }
    h1, h2, h3 {
        color: #4CAF50;
    }
    .stAlert {
        background-color: #2d2d2d;
        border: 1px solid #4CAF50;
        color: #e0e0e0;
        padding: 10px;
        border-radius: 5px;
    }
    .css-1d391kg {
        background-color: #252525;
    }
    .stRadio > label {
        color: #e0e0e0;
    }
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        color: #e0e0e0;
    }
    .streamlit-expanderHeader {
        background-color: #2d2d2d;
        color: #4CAF50;
    }
    .streamlit-expanderContent {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    .stFileUploader > div > button {
        background-color: #4CAF50;
        color: white;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    .stChatMessage {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #4CAF50;
    }
    .stMarkdown {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@st.cache_resource
def load_model():
    MODEL_URL = "https://drive.google.com/file/d/1rcGeyh5fW5tS0RVi6jHuMAnU7K9kSu/view?usp=sharing"  # Shareable link
    MODEL_PATH = "plant_disease_model.h5"

    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... This may take a moment.")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Failed to download model using gdown: {str(e)}")
            return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

model = load_model()
if model is None:
    st.stop()

disease_mapping = {
    0: "Tomato___Septoria_leaf_spot", 1: "Apple___Black_rot", 2: "Apple___Cedar_apple_rust", 
    3: "Apple___healthy", 4: "Blueberry___healthy", 5: "Cherry___Powdery_mildew", 
    6: "Cherry___healthy", 7: "Corn___Cercospora_leaf_spot Gray_leaf_spot", 
    8: "Corn___Common_rust_", 9: "Corn___Northern_Leaf_Blight", 10: "Corn___healthy", 
    11: "Grape___Black_rot", 12: "Grape___Esca_(Black_Measles)", 13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 
    14: "Grape___healthy", 15: "Orange___Haunglongbing_(Citrus_greening)", 
    16: "Peach___Bacterial_spot", 17: "Peach___healthy", 18: "Pepper___Bacterial_spot", 
    19: "Pepper___healthy", 20: "Potato___Early_blight", 21: "Potato___Late_blight", 
    22: "Potato___healthy", 23: "Raspberry___healthy", 24: "Soybean___healthy", 
    25: "Squash___Powdery_mildew", 26: "Strawberry___Leaf_scorch", 27: "Strawberry___healthy", 
    28: "Tomato___Bacterial_spot", 29: "Tomato___Early_blight", 30: "Tomato___Late_blight", 
    31: "Tomato___Leaf_Mold", 32: "Apple___Apple_scab", 33: "Tomato___Spider_mites", 
    34: "Tomato___Target_Spot", 35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus", 
    36: "Tomato___Tomato_mosaic_virus", 37: "Tomato___healthy"
}

genai.configure(api_key="AIzaSyCTN7ONXDJRINqBZd-Oldp4CR0HFAPwxBs")
chat = genai.GenerativeModel("gemini-1.5-flash").start_chat(history=[])

def get_gemini_response(question, language):
    try:
        prompt = f"Respond in {language}. {question}"
        response = chat.send_message(prompt)
        response.resolve()
        return response.text
    except Exception as e:
        st.error(f"Error getting Gemini response: {str(e)}")
        return None

def main():
    st.title("🌿 Smart Plant Disease Detection")
    st.subheader("Empowering Farmers with AI-Driven Insights for Healthier Crops")

    page = st.sidebar.selectbox("Navigate", ["Home", "About", "Disease Detection", "Chat with AI"])

    if page == "Home":
        display_home_page()
    elif page == "About":
        display_about_page()
    elif page == "Disease Detection":
        display_disease_detection_page()
    elif page == "Chat with AI":
        display_chat_page()

def display_home_page():
    st.write("Welcome to the Smart Plant Disease Detection System!")
    st.write("This innovative tool combines advanced image recognition with AI to help farmers identify and manage plant diseases effectively.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Features")
        st.write("- Instant disease detection from plant images")
        st.write("- AI-powered disease information and treatment advice")
        st.write("- Multi-language support for global accessibility")
        st.write("- Interactive chatbot for personalized guidance")
    
    with col2:
        st.subheader("Getting Started")
        st.write("1. Navigate to the 'Disease Detection' page")
        st.write("2. Upload or capture an image of your plant")
        st.write("3. Get instant disease diagnosis and information")
        st.write("4. Chat with our AI for more detailed advice")

    st.subheader("🌿 Targeted Diseases")
    st.write("""
    We are currently targeting the following 38 common plant diseases. 
    Please ensure your upload is from this list. Uploading images of diseases 
    not on this list will result in a warning message and no accurate diagnosis.
    """)
    
    important_diseases = [
        "Tomato___Septoria_leaf_spot", "Apple___Black_rot", 
        "Apple___Cedar_apple_rust", "Apple___healthy", 
        "Blueberry___healthy", "Cherry___Powdery_mildew", 
        "Cherry___healthy", "Corn___Cercospora_leaf_spot Gray_leaf_spot", 
        "Corn___Common_rust_", "Corn___Northern_Leaf_Blight", 
        "Corn___healthy", "Grape___Black_rot", 
        "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 
        "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", 
        "Peach___Bacterial_spot", "Peach___healthy", 
        "Pepper___Bacterial_spot", "Pepper___healthy", 
        "Potato___Early_blight", "Potato___Late_blight", 
        "Potato___healthy", "Raspberry___healthy", 
        "Soybean___healthy", "Squash___Powdery_mildew", 
        "Strawberry___Leaf_scorch", "Strawberry___healthy", 
        "Tomato___Bacterial_spot", "Tomato___Early_blight", 
        "Tomato___Late_blight", "Tomato___Leaf_Mold", 
        "Apple___Apple_scab", "Tomato___Spider_mites", 
        "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", 
        "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
    ]
    
    for disease in important_diseases:
        st.write(f"- {disease}")

    st.warning("**Warning:** Uploading images of diseases not in this list will lead to inaccurate results. Please ensure your uploads are from the targeted diseases.")

def display_about_page():
    st.header("About Smart Plant Disease Detection")
    
    st.subheader("🌟 Impact")
    st.write("""
    Our Smart Plant Disease Detection system is revolutionizing agriculture by:
    - Reducing crop losses through early disease detection
    - Minimizing pesticide use with targeted treatment recommendations
    - Empowering farmers with instant access to expert-level plant health information
    - Promoting sustainable farming practices worldwide
    """)
    
    st.subheader("🔮 Future Potential")
    st.write("""
    - Integration with IoT devices for automated monitoring and alerts
    - Expansion of the disease database to cover more crops and regions
    - Development of predictive models for disease outbreaks based on environmental data
    - Creation of a global network for real-time plant disease tracking and management
    """)
    
    st.subheader("💡 Innovation Level")
    st.write("""
    Our system stands at the forefront of agricultural technology by:
    1. Utilizing state-of-the-art deep learning models for accurate disease identification
    2. Incorporating large language models for context-aware treatment recommendations
    3. Offering a user-friendly interface accessible to farmers worldwide
    4. Continuously learning and improving through user interactions and feedback
    """)

def display_disease_detection_page():
    st.header("Plant Disease Detection")

    if 'predicted_disease' not in st.session_state:
        st.session_state.predicted_disease = None
    if 'disease_info' not in st.session_state:
        st.session_state.disease_info = {}

    st.subheader("Upload or Capture Image of Your Crop")
    image_source = st.radio("Select image source:", ("Upload Image", "Capture from Camera"))

    if image_source == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.image = image
            st.success("Image uploaded successfully.")
    else:
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            image = Image.open(camera_image)
            st.session_state.image = image
            st.success("Image captured successfully!")

    if 'image' in st.session_state and st.session_state.image is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Current Image")
            st.image(st.session_state.image, caption="Current Image", width=300)

        with col2:
            img_array = preprocess_image(st.session_state.image)
            prediction = model.predict(img_array)[0]
            predicted_class_index = np.argmax(prediction)
            predicted_probability = prediction[predicted_class_index]
            predicted_disease = disease_mapping.get(predicted_class_index, "Unknown Disease")

            st.session_state.predicted_disease = predicted_disease
            st.subheader("Prediction Result")
            st.info(f"Predicted Disease: {predicted_disease}")
            st.progress(float(predicted_probability))
            st.write(f"Confidence: {predicted_probability:.2f}")

        if st.session_state.predicted_disease != "Unknown Disease":
            language = st.selectbox("Select language for responses:", 
                                    ["English", "Spanish", "French", "German", 
                                     "Italian", "Chinese", "Japanese", 
                                     "Korean", "Hindi", "Arabic"])

            if st.button("Get Disease Information"):
                with st.spinner("Generating disease information..."):
                    description_query = f"Please describe the disease '{st.session_state.predicted_disease}'."
                    cures_query = f"What are the recommended cures for '{st.session_state.predicted_disease}'?"
                    medicines_query = f"What medicines can be used for treating '{st.session_state.predicted_disease}'?"

                    st.session_state.disease_info = {
                        "description": get_gemini_response(description_query, language),
                        "cures": get_gemini_response(cures_query, language),
                        "medicines": get_gemini_response(medicines_query, language),
                    }

        if st.session_state.disease_info:
            st.subheader(f"Disease Information ({language})")
            with st.expander("Description", expanded=True):
                st.write(st.session_state.disease_info["description"])
            with st.expander("Recommended Cures"):
                st.write(st.session_state.disease_info["cures"])
            with st.expander("Medicines"):
                st.write(st.session_state.disease_info["medicines"])

def display_chat_page():
    st.header("Chat with AI Assistant")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["bot"])

    user_input = st.chat_input("Ask about plant diseases or treatments...")

    if user_input:
        st.session_state.chat_history.append({"user": user_input, "bot": ""})

        if st.session_state.disease_info:
            context = f"""
            Disease: {st.session_state.predicted_disease}
            Description: {st.session_state.disease_info['description']}
            Cures: {st.session_state.disease_info['cures']}
            Medicines: {st.session_state.disease_info['medicines']}
            """
            response = get_gemini_response(f"Based on this information: {context}\n\nUser question: {user_input}", "English")
        else:
            response = get_gemini_response(f"The user is asking about plant diseases. User question: {user_input}", "English")

        st.session_state.chat_history[-1]["bot"] = response

        with st.chat_message("assistant"):
            st.write(response)

if __name__ == "__main__":
    main()
