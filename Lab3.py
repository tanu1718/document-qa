import streamlit as st
from openai import OpenAI

# Set up the title and sidebar for model selection
st.title("Lab 3 -- Chatbot")

openAI_model = st.sidebar.selectbox("Which Model?", ("mini", "regular"))

if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"

# Create an OpenAI client
if 'client' not in st.session_state:
    api_key = st.secrets["openai_api_key"]
    st.session_state.client = OpenAI(api_key=api_key)

# Set up the session state to hold messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Function to maintain a buffer of only the last two user messages and corresponding assistant responses
def maintain_message_buffer():
    # Only keep the last two user messages and corresponding assistant responses
    buffer_size = 4  # 2 user messages + 2 assistant responses
    if len(st.session_state["messages"]) > buffer_size:
        # Remove older messages
        st.session_state["messages"] = st.session_state["messages"][-buffer_size:]

# Display the messages in chat format
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# Get user input and send it to the model
if prompt := st.chat_input("What is up?"):
    # Append user input to the session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the user input in the chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the response from OpenAI
    client = st.session_state.client
    stream = client.chat.completions.create(
        model=model_to_use,
        messages=st.session_state.messages,
        stream=True
    )

    # Display assistant's response as it streams
    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    # Append assistant's response to the session state
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Maintain message buffer to only keep the last 2 messages (and corresponding assistant responses)
    maintain_message_buffer()