# Import the necessary libraries
import os
from dotenv import load_dotenv  
import streamlit as st
from openai import OpenAI  
from part3 import Head_Agent

st.title("Mini Project 2: Streamlit Chatbot")

# TODO: Replace with your actual OpenAI API key
load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

# Define a function to get the conversation history (Not required for Part-2, will be useful in Part-3)
def get_conversation() -> str:
    history = ""
    for message in st.session_state.messages:
        role = message["role"].capitalize()
        content = message["content"]
        history += f"{role}: {content}\n"
    return history

# Check for existing session state variables
if "head_agent" not in st.session_state:
    st.session_state.head_agent = Head_Agent(
        openai_key=os.getenv("OPENAI_API_KEY"),
        pinecone_key=os.getenv("PINECONE_API_KEY"),
        pinecone_index_name="ml-index-part3"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
# ... (code for displaying messages)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Wait for user input
prompt = st.chat_input("What would you like to chat about?")
if prompt:
    # ... (append user message to messages)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ... (display user message)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.status("Agent is thinking...", expanded=True) as status:
            st.write("Checking query safety...")
            
            # Update the agent's internal state
            st.session_state.head_agent.latest_user_query = prompt
            # Pass the current history so the Rewriter/Answerer can see it
            st.session_state.head_agent.history = st.session_state.messages[:-1]
            
            # Run the main loop
            full_response = st.session_state.head_agent.main_loop()
            
            status.update(label="Response generated!", state="complete", expanded=False)
        
        # Display the final answer
        st.markdown(full_response)

    # ... (append AI response to messages)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
