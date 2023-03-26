import streamlit as st
import random
import pandas as pd
import json
from copy import deepcopy





def display():
    f = open('data/extraction/doi_text.json')
    readin_data = json.load(f)
    readin_df = pd.read_csv('data/extraction/extract.csv')

    st.session_state['original data'] = readin_data
    st.session_state["option"] = deepcopy(readin_data)
    st.session_state['extract_df'] = readin_df
    st.title("Materials data extraction")
    with st.expander('', expanded=True):
        placeholder = st.empty()

    if "p2_initial" not in st.session_state:
        st.session_state["p2_initial"] = 1

    if st.session_state["p2_initial"] == 1:
        choose_txt()
        st.session_state["p2_initial"] = 0

    s1, s2 = st.columns([10,1])
    if s1.button('Next'):
        choose_txt()

    txt = st.session_state["txt"]["txt"]
    # st.code("This is a test", language="markdown")
    # placeholder.text_area(label="", value=txt, height=400)
    placeholder.markdown(txt, unsafe_allow_html=True)


    if s2.button("Extract"):
        extract()
    s1.image('src/UNSW.jpg', width=150)
    s2.image('src/g.jpg', width=60)

    s3, s4 = st.columns([3.5,1])
    s3.caption('Copyright 2023 University of New South Wales')
    s4.caption('Technical Support GreenDynamics Pty. Ltd')




def choose_txt():
    original_data = st.session_state['original data']
    data = st.session_state["option"]

    # random choose a txt
    doi = random.choice(list(data.keys()))
    txt = data[doi]
    del data[doi]

    if len(data.keys()) == 0:
        data = original_data
    #
    st.session_state["option"] = data
    st.session_state['doi'] = doi
    st.session_state["txt"] = {"txt": txt}

    return 0


def extract():
    df = st.session_state['extract_df']
    col = df.columns
    doi = st.session_state['doi']

    info = df[df['doi'] == doi]

    doi_cols = [i for i in col if 'doi' in i]
    stack_cols = [i for i in col if 'Substract' in i or 'ETL' in i or 'Perovskite' in i]
    htl_cols = [i for i in col if 'HTL' in i]
    back_cols = [i for i in col if 'Backcontact' in i]
    cell_cols = [i for i in col if 'Cell' in i or 'Module' in i]
    stability_cols = [i for i in col if 'Stability' in i]

    doi_df = info[doi_cols]
    stack_df = info[stack_cols]
    htl_df = info[htl_cols]
    back_df = info[back_cols]
    cell_df = info[cell_cols]
    stability_df = info[stability_cols]

    st.write('**Doi**')
    make_table(doi_df)
    st.write("**HTL Information**")
    make_table(stack_df)
    st.write("**Backcontact Information**")
    make_table(htl_df)
    st.write("**Stack Information**")
    make_table(back_df)
    st.write('**Cell Information**')
    make_table(cell_df)
    st.write('**Stability Information**')
    make_table(stability_df)

def make_table(df):
    ind = df.columns
    d = df.values
    table = pd.DataFrame(index=ind, data=d[0], dtype='string', columns=['Record'])
    # table = table.style.hide_index()
    # st.table(table)
    st.table(table.style.hide_columns())