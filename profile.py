import streamlit as st

def show():
    # Common CSS for consistent styling
    st.markdown(
        """
        <style>
            .main {
                background-color: #1e1e1e; /* Dark black shade */
 /* Ensure main content text color is white */
            }

            body {
                margin: 0;
                padding: 0;
                background-color: #000000; /* Set background color to black */
                color: black; /* Ensure text color is white for contrast */
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
            .main-header h1 {
                text-align: center;
                margin-top: 60px;
                color: #ad8aff;
            }
            .content {
                padding: 20px;
                margin-top: 20px;
                border-radius: 10px;
                background-color: #1e1e1e; /* Darker background for content */
            }
            .footer {
                text-align: center;
                margin-top: 50px;
                color: black;
            }
            .faq-section {
                background-color: #1a1a1a;
                padding: 30px;
                border-radius: 20px;
                margin-top: 50px;
                color: white; /* Set the text color to white */
            }
            .footer {
                background-color: #e6ccff;
                padding: 20px;
                text-align: center;
                margin-top: 50px;
            }
            .input-box {
                width: 50%;
                padding: 10px;
                border-radius: 25px;
                border: 2px solid #ad8aff;
                color: white;  /* Set the input text color to white */
            }
           
            body, h3, span {
                color: white; /* Set all text to white by default */
            }
            .faq-answer {
            color: white; /* Set answer text color to white */
        }
            .submit-button {
                background-color: #e6ccff; /* Lavender color */
                color: black; /* Change text color to black for contrast */
                border: none;
                padding: 10px 20px;
                border-radius: 25px;
                cursor: pointer;
            }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="main-header" style="text-align: center;">
            <h1 style="margin-bottom: 0; line-height: 1.2;">
                <span style="color: #ffffff;">Your</span>, 
                <span style="color: #9D4FDB;"><i>Profile.</i></span>
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="content">
            <p style="color: white;"><strong>Username</strong>: {username}</p>
            <p style="color: white;"><strong>Logged in</strong>: {status}</p>
        </div>
        """.format(username=st.session_state.username, status='Yes' if st.session_state.logged_in else 'No'),
        unsafe_allow_html=True
    )

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "Login"
        st.success("You have been logged out.")

    # Updated FAQ section with collapsible entries
    st.markdown(
        """
        <div class="faq-section">
            <h3>FREQUENTLY ASKED QUESTIONS</h3>
        """,
        unsafe_allow_html=True
    )

    faqs = [
        ("How do I log in?", "Enter your username and password on the login page."),
        ("What if I forget my password?", "Use the 'Forgot Password' link to reset it."),
        ("Can I switch accounts?", "Yes, simply log out and log back in with a different account."),
        ("What happens when I log out?", "You will be redirected to the login page."),
        ("How can I update my profile?", "Go to your profile settings to make changes."),
        ("Are my data secure?", "Yes, your data is protected and not shared with third parties."),
        ("Can I use the summarizer for private videos?", "No, only public videos can be summarized."),
        ("How long does it take to summarize a video?", "It typically takes just a few seconds."),
        ("Can I download the summary?", "Yes, summaries can be downloaded in PDF format."),
    ]

    for question, answer in faqs:
        with st.expander(question, expanded=False):  # Collapsible FAQ entry
             st.markdown(f'<div class="faq-answer">{answer}</div>', unsafe_allow_html=True)

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )  # Closing the FAQ section

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
                <p style="color : #000000;">© 2024 YouTube Summarizer. All rights reserved.</p>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
