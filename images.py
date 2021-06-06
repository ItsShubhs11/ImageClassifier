import streamlit as st
from fastai.vision.all import *



def load_img(path):
    image = Image.open(path)
    w, h = image.size
    dim = (500, int((h*500)/w))
    return image.resize(dim)
st.markdown("# Fruit Classifier")
st.markdown("Upload an image and the classifier will tell you whether its rotten, ripe or unripe fruit.")
file_bytes = st.file_uploader("Upload a file", type=("png", "jpg", "jpeg", "jfif"))
file_bytes = st.file_uploader("Upload a file", type=("png", "jpg", "jpeg", "jfif"))
