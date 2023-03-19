import streamlit as st

from pages.page2 import display as p2display
from pages.page4 import display as p4display

st.set_page_config(layout="wide")

with st.sidebar:
    add_radio = st.radio(
        "Green Dynamics",
        ("Materials Explorer", "BOXVIA")
    )


if add_radio == 'Materials Explorer':
    p4display()
elif add_radio == 'BOXVIA':
    p2display()


