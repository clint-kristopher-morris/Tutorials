import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

# Constants
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
TITLE = 'A.I. Super Resolution'
TITLE_PADDED = '&emsp;'*4 + TITLE
IM_CONSTANTS = {'LOGO': 'https://i.ibb.co/xhcsJ0d/duck-guy.png'}


# Load the model (Cached)
@st.cache_resource
def load_model():
    return hub.load(SAVED_MODEL_PATH)


# Convert image to tensor
def convert_image(raw_image):
    if raw_image.shape[-1] == 4:
        raw_image = raw_image[..., :-1]
    image_size = (tf.convert_to_tensor(raw_image.shape[:-1]) // 4) * 4
    cropped_image = tf.image.crop_to_bounding_box(raw_image, 0, 0, image_size[0], image_size[1])
    processed_image = tf.cast(cropped_image, tf.float32)
    return tf.expand_dims(processed_image, 0)


# Convert tensor to numpy array
def tensor2np(image_tensor):
    image_data = tf.clip_by_value(np.asarray(image_tensor), 0, 255)
    return Image.fromarray(tf.cast(image_data, tf.uint8).numpy())


# Upscale image using the model
def upscale_image(raw_im, model):
    im_tensor = convert_image(raw_im)
    upscaled_image = model(im_tensor)
    return tensor2np(tf.squeeze(upscaled_image))


# Main Streamlit app
def main():
    st.set_page_config(TITLE, page_icon=IM_CONSTANTS['LOGO'], layout='wide')
    st.title(TITLE_PADDED)
    m = load_model()

    # UI setup and information display
    _, col1, _, col2, _ = st.columns([2.5, 4.5, 1, 6, 2.5])

    with col1:
        st.markdown('## Empowering Clarity, Revealing Beauty with A.I.')
        st.markdown("""
            <style>
                .gray-container {
                    background-color: #f0f2f6;  /* Gray background */
                    border-radius: 10px;  /* Rounded corners */
                    padding: 20px;  /* Padding */
                    font-size:22px;
                }
            </style>
            <div class="gray-container">
                <img src="https://i.ibb.co/xhcsJ0d/duck-guy.png" alt="Example Image" style="vertical-align: middle;  width:24px;">&emsp;1,453,443 Images Improved
            </div>
            """,
            unsafe_allow_html=True
            )
        # white space
        for _ in range(4):
            st.markdown("")
        # place for logo
        logo1, logo2, _ = st.columns([1, 1, 2])
        logo1.image(IM_CONSTANTS['LOGO'], width=110)



    with col2:
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            raw_image = tf.convert_to_tensor(np.array(image))
            upscaled_image = upscale_image(raw_image, m)
            st.image(np.array(upscaled_image), caption='Upscaled Image', use_column_width=True)


if __name__ == "__main__":
    main()
