import streamlit as st
from api_client import query_backend  
from config import APP_TITLE, LOGO_PATH

st.set_page_config(page_title=APP_TITLE, layout="wide")

st.image(LOGO_PATH, width=150)
st.title(APP_TITLE)

# Get user input
user_question = st.text_input("Ask a question about tourism in Uruguay:")

if user_question:
    # Get response from backend
    response = query_backend(user_question)
    st.write(response)
