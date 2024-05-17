import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from IPython.display import display
import matplotlib as mpl
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.image as img
from tensorflow.keras.models import load_model

from util import set_background, set_sidebar_background
import brain_tumor
import pneumonia



def login():
    placeholder = st.empty()

    actual_user_name = "amit"
    actual_password = "12345"

    # Insert a form in the container
    with placeholder.form("login"):
        st.markdown(":blue[ Enter your credentials]")
        user_name = st.text_input("User Name")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit and user_name == actual_user_name and password == actual_password:
        st.session_state["logged_in"] = True
        placeholder.empty()
    elif submit and user_name != actual_user_name and password != actual_password:
        st.error("Login failed")
    else:
        pass


# initial page setup
st.set_page_config(
    page_title="Medical Image Analysis",
    page_icon="👨‍⚕️"
    # layout="wide",
    # initial_sidebar_state="expanded"
)

# Background Image set

# set_background('./background/background.jpg')
# set_sidebar_background('./background/side-bar.jpg')

# website header
st.header(':blue[Welcome to our Medical Image Analysis website.] :syringe: :pill: :ambulance:', divider='rainbow')
    # st.('_Streamlit_ is :blue[cool] :sunglasses:')


# Login Page
if "logged_in" in st.session_state:
    pass
else:
    login()

# choose prediction type
if "logged_in" in st.session_state:
    # with st.sidebar.container(border=True):
    st.sidebar.title(":orange[Select Disease Detection Option]")
    st.markdown("""
    <style>
    .stSelectbox:first-of-type > div[data-baseweb="select"] > div {
        background-color: #F9F582;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    option = st.sidebar.selectbox("Disease Name", ["Brain Tumor Detection", "Pneumonia Detection"], key="option1")
    
    if option == "Brain Tumor Detection":
        brain_tumor.prediction()
    elif option == "Pneumonia Detection":
        pneumonia.prediction()

