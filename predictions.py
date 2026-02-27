import streamlit as st
from os import path

DATA_DIR = 'data_files/'
st.image(path.join(DATA_DIR, 'logo.png'), width=450)
st.set_page_config(page_title="Tennis Predictions", layout="centered")

st.title("Tennis Betting Predictions")

st.write("This is the base page for the tennis predictions app. Add your content here and deploy to Streamlit Cloud.")
