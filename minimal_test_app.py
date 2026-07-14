import streamlit as st
from sentence_transformers import SentenceTransformer

st.write("Before model")
print("🚀 [MINIMAL TEST] About to load SentenceTransformer...", flush=True)
model = SentenceTransformer("all-MiniLM-L6-v2")
print("🚀 [MINIMAL TEST] SentenceTransformer LOADED successfully!", flush=True)
st.write("After model")
