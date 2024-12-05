import streamlit as st

class MultiPage:
    """Framework for combining multiple streamlit applications
    """
    def __init__(self) -> None:
        self.pages = []
    
    def add_page(self, title, func):
        self.pages.append(
            {
            'title': title,
            'function': func
            }
        )
    
    def run(self):
        #st.sidebar.title("Grind Master")
        page = st.logo("bisalloy.png")
        page = st.sidebar.image("amman.png", width=120)
        page = st.sidebar.header("PT AMMAN Batu Hijau Mine")
        page = st.sidebar.header(":rainbow[Mill Liner Wear Reporting App]")
        page = st.sidebar.image("mill.png", width=240)
        page = st.sidebar.header("***Next Gen Grinding Intelligence***", divider='gray')
        page = st.sidebar.markdown("###")
        page = st.sidebar.radio(
            'App Navigation', 
            self.pages,
            format_func=lambda page: page['title']  # Function to modify the display of the labels.

        )
        
        page['function']()
        page = st.sidebar.markdown("###")
        page = st.sidebar.markdown("###")
        page = st.sidebar.markdown("###")
        page = st.sidebar.markdown("###")
        page = st.sidebar.markdown("###")
        page = st.sidebar.caption("Developed by Bisalloy Digital Solutions 2024")
        page = st.sidebar.caption("Contact us@ www.bisalloy.com.au")
        page = st.sidebar.caption("Email us@ charles.curry@bisalloy.com.au")