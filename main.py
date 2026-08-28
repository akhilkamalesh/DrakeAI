import streamlit as st
import requests

st.set_page_config(page_title="Drake RAG Agent", page_icon="🎵")
st.title("🎵 Drake Lyric & Audio Intelligence")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user input
if prompt := st.chat_input("Ask about Drake lyrics, vibes, or audio features..."):
    # Display user message in UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Searching stanzas and analyzing track audio..."):
            try:
                response = requests.post(
                    "http://localhost:8000/api/chat",
                    json={"prompt": prompt, "history": st.session_state.messages[:-1]}
                )
                answer = response.json().get("response", "No response received.")
            except Exception as e:
                answer = f"Error connecting to backend: {e}"
            
            st.markdown(answer)
    
    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})