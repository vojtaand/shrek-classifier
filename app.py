import pathlib
import sys

import streamlit as st
from fastai.vision.all import PILImage, load_learner
from huggingface_hub import hf_hub_download

# A model exported in Colab (Python 3.13) then loads on older Pythons too.
sys.modules.setdefault("pathlib._local", pathlib)

REPO_ID = "VojtaAnd/classifier-shrek"


@st.cache_resource          # load the model only once, not on every click
def load_model():
    return load_learner(hf_hub_download(REPO_ID, "model.pkl"))


st.title("Character from Shrek classifier")
st.write("Upload a photo of a character from Shrek and the model will tell you which one it is")

learn = load_model()
uploaded = st.file_uploader("Your image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    st.image(uploaded, width=300)
    prediction, index, probabilities = learn.predict(PILImage.create(uploaded.getvalue()))
    st.subheader(f"Prediction: {prediction}")
    for category, probability in zip(learn.dls.vocab, probabilities):
        st.progress(float(probability), text=f"{category}: {float(probability):.1%}")
