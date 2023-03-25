import streamlit as st

from pages.material_extraction import display as p1display
from pages.material_explorer import display as p2display
from pages.BOXVIA import display as p3display


st.set_page_config(layout="wide")

with st.sidebar:
    add_radio = st.radio(
        "Green Dynamics",
        ("Material Data Extraction", "Materials Explorer", "BOXVIA")
    )


if add_radio == 'Materials Data Extraction':
    p2display()
elif add_radio == 'Materials Explorer':
    p2display()
elif add_radio == 'BOXVIA':
    p3display()


