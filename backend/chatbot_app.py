import streamlit as st
import backend.Kranos as Kranos

st.title("Ebot")
user_input = st.text_input("User Input")
submit_button = st.button("Submit")

if submit_button:
    
    chatbot_response = Kranos.predict_response(user_input)
    st.text(chatbot_response)
