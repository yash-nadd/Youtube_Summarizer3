import streamlit as st
st.set_page_config(layout="wide")
from streamlit_navigation_bar import st_navbar
import home, tools, blog, about
import yaml
import os
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi



def load_config():
    try:
        with open('LoginInfo/config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error("Configuration file not found.")
        return None
    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML file: {e}")
        return None

def save_config(config):
    with open('LoginInfo/config.yaml', 'w') as file:
        yaml.safe_dump(config, file)


config = load_config()
if config is None:
    st.stop()

credentials = config.get('credentials', {})
if 'usernames' not in credentials:
    st.error("The 'usernames' key is missing in the credentials.")
    st.stop()


def authenticate(username, password):
    if username in credentials['usernames']:
        stored_password = credentials['usernames'][username].get('password')
        if stored_password == password:
            return True
    return False


def register_user(username, password):
    if username in credentials['usernames']:
        return False  
    credentials['usernames'][username] = {'password': password}
    save_config(config)
    return True

def chatbot(query):
    responses = {
    "what does this app do?": "This app summarizes YouTube videos. Simply enter a YouTube URL and get a concise summary.",
    "how do i use the youtube summarizer?": "Enter a YouTube video URL and press the 'Summarize' button. The summary will be displayed below the video.",
    "can i download the summary?": "Yes, you can download the generated summary as a PDF.",
    "what formats are supported?": "Currently, we support YouTube video links for summarization.",
    "how long does it take to generate a summary?": "The time taken depends on the video length, but it typically takes a few seconds.",
    "is there a limit to video length?": "There's no strict limit, but very long videos may take longer to summarize.",
    "can i use this app for private videos?": "No, this app only supports public YouTube videos.",
    "what if the summary is not accurate?": "The summary is generated using AI; you can always refer to the video for complete context.",
    "who developed this app?": "This app was developed as a project to demonstrate YouTube video summarization.",
    "hello": "Hi there! How can I assist you today?",
    "hi": "Hello! What can I do for you?",
    "how are you?": "I'm just a bot, but I'm here to help you! What do you need?",
    "thank you": "You're welcome! If you have any more questions, feel free to ask!",
    "goodbye": "Goodbye! Have a great day!",
    "help": "Sure! Just ask me anything about using the YouTube summarizer."
}

    return responses.get(query.lower(), "I don't understand the question. Please ask something else.")


def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.show_chatbot = False  
        st.session_state.chat_history = []  

    if not st.session_state.logged_in:
        with st.sidebar:
            st.title("Login / Register")

            tab = st.selectbox("Select option", ["Login", "Register"])

            if tab == "Login":
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit_button = st.button("Login")

                if submit_button:
                    if authenticate(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.page = "Home"  
                    else:
                        st.sidebar.error("Incorrect username or password.")
            
            elif tab == "Register":
                reg_username = st.text_input("New Username")
                reg_password = st.text_input("New Password", type="password")
                reg_button = st.button("Register")

                if reg_button:
                    if register_user(reg_username, reg_password):
                        st.sidebar.success("Registration successful! You can now log in.")
                    else:
                        st.sidebar.error("Username already exists. Choose a different username.")
    else:
        col1, col2 = st.columns([9, 1])
        with col2:
            chatbot_button = st.button("💬 Chat", key="chatbot_button", help="Click to open chatbot")

        pages = ["Home", "Tools", "Blog", "About", "Profile"]
        page = st_navbar(pages)

        if page == "Home":
            home.show()
        elif page == "Tools":
            tools.show()
        elif page == "Blog":
            blog.show()
        elif page == "About":
            about.show()
       

        if chatbot_button:
            st.session_state.show_chatbot = not st.session_state.show_chatbot

        if st.session_state.show_chatbot:
            with st.expander("💬 Chatbot", expanded=True):
                st.markdown(
                    """
                    <style>
                    .chatbot-container {
                        max-width: 300px;  /* Adjust the width */
                    }
                    </style>
                    """, unsafe_allow_html=True
                )
                st.write("### Chat with our bot!")
                
                user_input = st.text_input("Ask something about this app:", key="chat_input", help="Type your question here")
                if user_input:
                    answer = chatbot(user_input)
                    st.session_state.chat_history.append(f"You: {user_input}")
                    st.session_state.chat_history.append(f"Bot: {answer}")
                    for line in st.session_state.chat_history:
                        st.write(line)

if __name__ == "__main__":
    main()