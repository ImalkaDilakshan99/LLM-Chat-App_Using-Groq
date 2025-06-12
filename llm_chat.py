import streamlit as st
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Create Groq LLM instance
llm = Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

st.title("🔍 Ask LLaMA3 (Groq)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything!"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = llm.complete(prompt)  # This uses Groq

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
