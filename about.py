import streamlit as st

def show():
    st.markdown(
        """
        <style>
            * {
                box-sizing: border-box; /* Include padding and border in width calculations */
            }
            body {
                margin: 0;
                padding: 0;
                background-color: #000; /* Set background color to black */
                color: white; /* Ensure text color is white for contrast */
            }
            .main {
                 background-color: #1e1e1e; /* Dark black shade */
                 color: white; /* Ensure main content text color is white */
                }

            .navbar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                background-color: #000;
                padding: 10px;
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
            h3, span {
            color: white; /* Set all text to white by default */
        }
            h1 {
                text-align: center;
                margin-top: 60px;
                color: #9D4FDB; /* Heading color */
            }
            .content {
                padding: 20px;
                margin-top: 100px; /* Adjust for fixed navbar */
                border-radius: 10px;
                background-color: #1e1e1e; /* Content background color */
                color: white; /* Text color in content */
            }
            .faq-section {
                background-color: #1a1a1a; /* FAQ section background */
                padding: 30px;
                border-radius: 20px;
                margin-top: 50px;
                color: white; /* Text color in FAQ */
            }
            .footer {
                background-color: #e6ccff;
                padding: 20px;
                text-align: center;
                margin-top: 50px;
                color: black; /* Adjust for contrast against light background */
            }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="main-header" style="text-align: center;">
            <h1 style="margin-bottom: 0; line-height: 1.2;">
                <span style="color: #ffffff;">About</span>, 
                <span style="color: #9D4FDB;"><i>Youtube Summarizer.</i></span>
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="content">
            The YouTube Summarizer is a tool designed to automatically extract and summarize the transcript of a YouTube video,
            providing a brief overview of the video's content.
            <br><br>
            Built using Python and Streamlit, this tool integrates various APIs and machine learning, deep learning, and natural language processing models to generate concise video summaries.
        </div>
        """, unsafe_allow_html=True
    )

 
    st.markdown("<div class='faq-section'><h3 >FREQUENTLY ASKED QUESTIONS</h3>", unsafe_allow_html=True)

    faqs = [
        ("Can I summarize any video?", "Yes, as long as it's public."),
        ("How long does it take?", "Just a few seconds to get the summary."),
        ("Can I save the summary?", "Yes, download it as a PDF.")
    ]

    for question, answer in faqs:
        with st.expander(question, expanded=False):  
            st.write(answer)

    st.markdown("</div>", unsafe_allow_html=True)  

    st.markdown(
        """
        <div class="footer">
            <p> 
                <a href="mailto:your-email@example.com" style="color: #000000;">Email</a> | 
                <a href="https://www.linkedin.com/in/your-linkedin-profile" target="_blank" style="color: #000000;">LinkedIn</a> | 
                <a href="https://twitter.com/your-twitter-handle" target="_blank" style="color: #000000;">Twitter</a> | 
                <a href="https://www.instagram.com/your-instagram-handle" target="_blank" style="color: #000000;">Instagram</a>
                <br>
                <br>
                <p style="color: #000000;">© 2024 YouTube Summarizer. All rights reserved.</p>
            </p>
        </div>
        """, unsafe_allow_html=True
    )
