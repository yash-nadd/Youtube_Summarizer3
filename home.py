import streamlit as st
import subprocess
import os
import tempfile
import shutil
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi

def show():
    st.markdown(
        """
        <style>
            * {
                box-sizing: border-box;
            }
            body {
                margin: 0;
                padding: 0;
            }
            .navbar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                background-color: #000;
                padding: 10px;
                width: 100%;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 100;
            }
            .navbar a {
                color: white;
                margin: 0 15px;
                text-decoration: none;
            }
            .navbar a:hover {
                color: #ad8aff;
            }
            .main-header {
                text-align: center;
                margin-top: 50px;
            }
            .main {
                background-color: #1e1e1e;
                padding-top: 60px;
            }
            .input-section {
                display: flex;
                justify-content: center;
                margin-top: 30px;
            }
            .input-box {
                width: 50%;
                padding: 10px;
                border-radius: 25px;
                border: 2px solid #ad8aff;
                color: white;
                background-color: #1a1a1a;
            }
            .input-box::placeholder {
                color: grey;
            }
            .centered-title {
                text-align: center;
                color: white;
                font-weight: bold;
                font-size: 24px;
                margin-top: 20px;
            }
            .audience-boxes {
                display: flex;
                justify-content: space-evenly;
                margin-top: 50px;
            }
            .audience-box {
                background-color: #e6ccff;
                padding: 20px;
                text-align: center;
                border-radius: 20px;
                width: 200px;
                color: black;
            }
            .faq-section {
                background-color: #1a1a1a;
                padding: 30px;
                border-radius: 20px;
                margin-top: 50px;
                color: white;
            }
            .footer {
                background-color: #e6ccff;
                padding: 20px;
                text-align: center;
                margin-top: 50px;
                color: black;
            }
            body, h1, h2, h3, h4,span,div {
                color: white;
            }
            .faq-answer {
                color: white; /* Set answer text color to white */
            }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="main-header" style="text-align: center;">
            <h1 style="margin-bottom: 0; line-height: 1.2;">
                <span style="color: #ffffff;">Watch less</span>, 
                <span style="color: #9D4FDB;"><i>understand more.</i></span>
            </h1>
            <h4 style="margin-top: 0; color: #FFFFFF;">Summarize YouTube videos in seconds!</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "summary" not in st.session_state:
        st.session_state.summary = ""

    col1, col2 = st.columns([2, 1])

    with col1:
        video_url = st.text_input("Enter YouTube URL", value=st.session_state.get('video_url', ''))

    with col2:
        summary_length = st.number_input("More words, More precise", min_value=10, max_value=10000, value=100, step=10)

    upload_file = st.file_uploader("Or upload a video file", type=["mp4", "avi", "mov"])

    submit_button = st.button("Submit", key="submit_button", help="Generate Summary")

    if submit_button:
        if video_url:
            st.session_state.video_url = video_url
            with open('texts/video_id.txt', 'w') as file:
                file.write(video_url)

            st.write("Generating summary... Please wait.")
            try:
                subprocess.run(["python3", "summarizer.py", str(summary_length)], check=True)
            except subprocess.CalledProcessError as e:
                st.error(f"Error occurred while generating summary: {e}")
                st.stop()

        elif upload_file:
            # Create a temporary directory to store the uploaded video
            with tempfile.TemporaryDirectory() as tmpdirname:
                video_path = os.path.join(tmpdirname, upload_file.name)
                with open(video_path, "wb") as f:
                    f.write(upload_file.getbuffer())

                st.write("Generating summary from the uploaded video... Please wait.")
                try:
                    subprocess.run(["python3", "video_summarizer.py", str(summary_length), video_path], check=True)
                except subprocess.CalledProcessError as e:
                    st.error(f"Error occurred while generating summary: {e}")
                    st.stop()

        if os.path.exists('texts/summary.txt'):
            with open('texts/summary.txt', 'r') as file:
                summary = file.read()

            st.session_state.summary = summary

        else:
            st.write("Summary not available yet.")

    if st.session_state.summary:
        st.subheader("Summary of the video:")
        st.markdown(
            f"""
            <div style="border: 2px solid #e6e6e6; padding: 20px; border-radius: 10px; display: flex; justify-content: space-between;">
                <div style="flex: 1; padding-right: 20px;">
                    {st.session_state.video_url and f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/{st.session_state.video_url.split("v=")[-1]}" frameborder="0" allowfullscreen></iframe>'}
                </div>
                <div style="flex: 2;">
                    <h4>Summary</h4>
                    <p>{st.session_state.summary}</p>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    # (Remaining sections of your code stay the same...)

show()
