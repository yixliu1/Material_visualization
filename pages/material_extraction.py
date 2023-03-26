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
    t1, t2 = st.columns([1, 0.2])
    st.image('src/title.png', width=150)
    st.title("Pervoksite Solar Cell FAIR Dataset Demo")
    st.write('Knowledge frame from: https://www.nature.com/articles/s41560-021-00941-3.')
    st.write('All examples are perovskite-related papers published after 2021.03. Records are exactly as our fine-tuned GPT-3 model.')
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
    st.image('src/logo.png', width=300)
    # s5, s6 = st.columns([1, 4.5])
    # s5.image('src/UNSW.png', width=150)
    # s6.image('src/green.png', width=200)

    s3, s4 = st.columns([2.5,1])
    url = 'http://www.greendynamics.com.au'
    # st.caption('**Copyright** 2023 University of New South Wales | **Technical Support** [GreenDynamics Pty. Ltd](%s)', url)
    # s4.caption('')
    st.caption(f"**Copyright** 2023 University of New South Wales | **Technical Support** [GreenDynamics Pty. Ltd]({url})")




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
    # st.write(info)

    doi_cols = [i for i in col if 'doi' in i]
    stack_cols = [i for i in col if 'Substrate' in i or 'ETL' in i or 'Perovskite' in i
                  or 'HTL' in i or 'Backcontact' in i]
    cell_cols = [i for i in col if 'Cell' in i or 'Module' in i]
    stability_cols = [i for i in col if 'Stability' in i]

    doi_df = info[doi_cols]
    stack_df = info[stack_cols]
    cell_df = info[cell_cols]
    stability_df = info[stability_cols]

    st.write('**Paper Link**')
    table = pd.DataFrame(index=['doi'], data=list(doi_df.values)[0][0].replace('.json', ''), dtype='string', columns=['Record'])
    st.table(table.style.hide_columns())
    st.write("**Stack & Synthesis Information**")
    make_table(stack_df)
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