import streamlit as st
from os import listdir, walk
from os.path import isfile, join
import pandas as pd
import json


def display():
    l2_clean, l2_soiling, l3_clean, l3_soiling, abbr_dict, l12_path, l123_path, l12_files, l123_files = read_data()

    st.header("Materials Explorer")
    l1 = st.selectbox("Select a functional coating", ('', 'Self-cleaning', 'Anti-soiling'))
    if l1:
        if l1 == 'Self-cleaning':
            l2_select = l2_clean
            l3_select = l3_clean
        elif l1 == 'Anti-soiling':
            l2_select = l2_soiling
            l3_select = l3_soiling
        l2 = st.selectbox("Select a property", l2_select)
        if l2:
            l3 = st.selectbox("Select a characterization", l3_select)

            if l3:
                # only l1
                if l2 == 'N/A' and l3 == 'N/A':
                    p = '12'
                    name = f'{abbr_dict[l1]}_'
                # l1 & l2
                if l2 != 'N/A' and l3 == 'N/A':
                    p = '12'
                    name = f'{abbr_dict[l2]}'
                # l1 & l3
                if l2 == 'N/A' and l3 != 'N/A':
                    p = '123'
                    name = f'{abbr_dict[l3]}_{abbr_dict[l1]}'
                # l1, l2 & l3
                if l2 != 'N/A' and l3 != 'N/A':
                    p = '123'
                    name = f'{abbr_dict[l3]}_{abbr_dict[l2]}'
                # st.write(name)
                if p == '12':
                    name = name.capitalize()
                    ff = [s for s in l12_files if name in s]
                    if len(ff) >= 1:
                        li = []
                        fp = f'{l12_path}/{ff[0]}'
                        with open(f'{fp}', 'r', encoding='utf-8') as file:
                            for line in file:
                                li.append(
                                    line.strip().split('\t'))
                        list_df = pd.DataFrame(li, columns=['index', 'Name', 'Total', 'Information', 'Potential'])
                        table = list_df.iloc[:,1:]

                        st.dataframe(table)
                    else:
                        st.write('There is current no available dataset.')
                else:
                    if l3 == 'Water contact angle':
                        # l123_files += 'angle_update'
                        columns = ['Material', 'Contact angle', 'Source']
                    elif l3 == 'Water sliding angle':
                        # l123_files += 'angle_update'
                        columns = ['Material', 'Sliding angle', 'Source']
                    elif l3 == 'Refractive index':
                        # l123_files += 'index_update'
                        columns = ['Material', 'Index', 'Source']
                    elif l3 == 'Transmittance':
                        # l123_files += 'transmittance_update'
                        columns = ['Material', 'Transmittance(%)', 'Source', 'Single/Double']
                    elif l3 == 'Durability':
                        # l123_files += 'durability'
                        columns = ['Material', 'Rate', 'Source']

                    ff = [s for s in l123_files if name in s]
                    if len(ff) >= 1:
                        fp = f'{ff[0]}'
                        # table = pd.read_excel(fp)
                        table = json_to_df(fp, columns)
                        range_col = table.columns[1]
                        slider = st.slider('Select a range of index', min(table[range_col]), max(table[range_col]),
                                           (min(table[range_col]), max(table[range_col])))
                        table = table[(slider[0]<=table[range_col])
                                      & (slider[1]>=table[range_col])]

                    # st.write(fp)
                    if len(ff) >= 1:
                        if 'Transmittance(%)' in columns:
                            c1, c2, c3 = st.columns([2, 0.5, 0.5])
                            sin = c2.checkbox('Single')
                            dou = c3.checkbox('Double')
                            if sin and dou:
                                pass
                            elif sin:
                                table = table[table['Single/Double']=='single']
                            elif dou:
                                table = table[table['Single/Double']=='double']

                        st.dataframe(table)
                    else:
                        st.write('There is current no available dataset.')


@st.cache(suppress_st_warning=True)
def read_data():
    l2_clean = ['', 'N/A', 'Hydrophobic/Superhydrophobic', 'Oleophobic/Surperoleophobic', 'Omniphobic/Amphiphobic',
                'Hydrophilic/Superhydrophilic', 'Photocatalytic']
    l2_soiling = ['', 'N/A', 'Antistatic']
    l3_clean = ['', 'N/A', 'Water contact angle', 'Water sliding angle', 'Refractive index', 'Transmittance', 'Durability']
    l3_soiling = ['', 'N/A', 'Refractive index', 'Transmittance', 'Durability']
    abbr_dict = {'Self-cleaning': 'self-cleaning', 'Anti-soiling': 'anti-soiling',
                 'Hydrophobic/Superhydrophobic': 'hydrophobic', 'Oleophobic/Surperoleophobic': 'oleophobic',
                 'Omniphobic/Amphiphobic': 'omniphobic', 'Hydrophilic/Superhydrophilic': 'hydrophilic',
                 'Photocatalytic': 'photocatalytic', 'Antistatic': 'antistatic',
                 'Water contact angle': 'contact_angle', 'Water sliding angle': 'sliding_angle',
                 'Refractive index': 'index', 'Transmittance': 'trans', 'Durability': 'dur'
                 }
    path = 'data/explore'
    # path = 'data/SC visulization dataset'
    l12_path = f'{path}/level1 only'
    l123_path = f'{path}/level1_level3 or level1_level2_level3'
    l12_files = [f for f in listdir(l12_path) if isfile(join(l12_path, f))]

    l123_files = []
    for root, dirs, files in walk(l123_path):
        for filename in files:
            if isfile(join(root, filename)):
                l123_files.append(join(root, filename))
    # l123_files = [f for f in listdir(l123_path) if isfile(join(l123_path, f))]

    return l2_clean, l2_soiling, l3_clean, l3_soiling, abbr_dict, l12_path, l123_path, l12_files, l123_files


def json_to_df(fp, columns):
    with open(fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    if 'Rate' in columns:
        for i in data:
            li = data[i]
            rate = li['rate']
            for j in li:
                if j != 'rate':
                    info = li[j]
                    for k in info:
                        rows.append([i, rate, k])
    elif 'Transmittance(%)' in columns:
        for i in data:
            li = data[i]
            for j in li:
                for k in li[j]:
                    rows.append([i, j, k[:-1], k[-1]])
    else:
        for i in data:
            li = data[i]
            for j in li:
                rows.append([i, j, li[j]])
    df = pd.DataFrame(rows, columns=columns)
    df[df.columns[1]] = df[df.columns[1]].astype(float)
    return df





