import streamlit as st
from multipage import MultiPage
from spages import SAG1, SAG2

MAGE_EMOJI_URL = "streamlitbis.png"
st.set_page_config(page_title='OptiWear®',page_icon=MAGE_EMOJI_URL, initial_sidebar_state = 'expanded', layout="centered")
#page_icon = favicon,

st.markdown(
            f"""
            <style>
                .reportview-container .main .block-container{{
                    max-width: 1500px;
                    padding-top: 0rem;
                    padding-right: 1rem;
                    padding-left: 1rem;
                    padding-bottom: 0rem;
                }}
    
            </style>
            """,
            unsafe_allow_html=True,
        )



app = MultiPage()

# add applications
app.add_page('🔵  SAG Mill #1', SAG1.app)
app.add_page('🟢  SAG Mill #2', SAG2.app)
#app.add_page('🟠  SuperVortex DE | Crusher On/Off', SVDEConoff.app)



# Run application
if __name__ == '__main__':
    app.run()
