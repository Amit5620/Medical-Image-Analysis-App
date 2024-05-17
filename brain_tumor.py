import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
# from IPython.display import display
import matplotlib as mpl
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.image as img
from tensorflow.keras.models import load_model



#Define Some Functions for prediction :

last_conv_layer_name = "Top_Conv_Layer"

def get_img_array(img, size = (224 , 224)):
    image = np.array(img)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    resized_image = cv2.resize(image, (224,224))
    resized_image = resized_image.reshape(-1,224,224,3)
    resized_image = np.array(resized_image)
    return resized_image
    # img = keras.utils.load_img(img_path, target_size=size)
    # array = keras.utils.img_to_array(img)
    # array = np.expand_dims(array, axis=0)
    # return array


def make_gradcam_heatmap(img_array, model , last_conv_layer_name = last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4 , view = False):
    # Load the original image
    img = np.array(img)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # img = keras.utils.load_img(img_path)
    # img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)


        
     
def decode_predictions(preds):
    classes = ['Glioma' , 'Meningioma' , 'No Tumor' , 'Pituitary']
    prediction = classes[np.argmax(preds)]
    return prediction



def make_prediction (img , model, last_conv_layer_name = last_conv_layer_name , campath = "cam.jpeg" , view = False):
    image = get_img_array(img)
    img_array = get_img_array(img, size=(224 , 224))
    preds = model.predict(img_array)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    save_and_display_gradcam(img, heatmap , cam_path=campath , view = view)
    return [campath , decode_predictions(preds)]
        




# Prediction base function
def prediction():
    st.header(':yellow[Brain Tumor Detection]:male-doctor:', divider='rainbow')


    placeholder = st.empty()

    with placeholder.form("upload"):
        st.header(':yellow[Provide a Brain X-ray image.]:mag:', divider='orange')
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        submit = st.form_submit_button("Predict")

    if submit and uploaded_file:
        placeholder.empty()
        image = Image.open(uploaded_file)
        # st.image(image, caption="Upload Image")

        # Load the Model
        model = load_model("./models/brain_tumor_prediction.h5")


        # Prediction
        campath, prediction = make_prediction(image, model, campath="123.jpeg", view=False)


        # Show the result
        with st.container(border=True):
            col1, col2 = st.columns([2,1])
            with col1:
                st.header(f':mag: :rainbow[Prediction: {prediction}]')
                st.write(f'You have {prediction} in your brain.')
            with col2:
                st.header(':orange[Total Types]')
                # st.write('1. No Tumor')
                # st.write('2. Glioma')
                # st.write('3. Meningioma')
                # st.write('4. Pituitary')

                c1, c2 = st.columns(2)
                with c1:
                    st.write('1. No Tumor')
                    st.write('3. Meningioma')
                with c2:
                    st.write('2. Glioma')
                    st.write('4. Pituitary')

        # cont = st.container(height=100, border=True)
        # cont.header(prediction)

        # col1, col2 = st.columns(2)
        # with col1:
        #     st.image(image, caption="Uploaded Image")
        
        # with col2:
        #     test_img = img.imread(campath)
        #     st.image(test_img, caption="Predicted Image")

        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image")
            
            with col2:
                test_img = img.imread(campath)
                st.image(test_img, caption="Predicted Image")

        with open('./123.jpeg', "rb") as file:
            btn = st.download_button(
                    label="Download Predicted Image",
                    data=file,
                    file_name="prediction.jpeg",
                    mime="image/jpeg"
                )
