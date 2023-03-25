import streamlit as st
import pandas as pd
import streamlit_toggle as toggle
from src.visualize.visualize1D import visualize as visual1D
from src.visualize.visualize2D import visualize as visual2D
from src.visualize.visualize import visualize as visual
import numpy as np
from plotly.subplots import make_subplots

"""
1. refresh the whole page when button
2. slow when 3d graph
3. load issue
"""

# @st.cache()
# def read_csv(uploaded_file):
#     df = pd.read_csv(uploaded_file)
#
#     st.session_state.df = df
#     return 0

@st.cache()
def delete_bottom(df):
    st.session_state
    return 0


def display():
    # st.image('src/title.png')
    t1, t2 = st.columns([0.1, 1])
    t1.image('src/g.jpg')
    t2.title("BOXVIA")
    st.text("Bayesian Optimization EXecutable and VIsualizable Application")

    st.subheader("Import Data")
    uploaded_file = st.file_uploader("Choose a file", help=".csv format")
    # uploaded_file = 'data/samples/Sample(d=5).csv'
    if uploaded_file is not None:
        # st.write(uploaded_file)
        table = run_parameter(uploaded_file)
        suggest_table(table)
        visualization_parameter()
    return 0


def run_parameter(uploaded_file):
    global vis
    kernel_dic = {'Exponential': 'Exponential', 'Linear': 'Linear', 'Matern 3/2': 'Matern32', 'Matern 5/2': 'Matern52',
                  'Radial Basis Function (RBF)': 'RBF', 'Rational Quadratic': 'RatQuad', 'Standard Period': 'StdPeriodic'}

    df = pd.read_csv(uploaded_file)
    if 'df' not in st.session_state or not df.equals(st.session_state.df):
        if 'run_record' in st.session_state:
            del st.session_state['run_record']
        if 'graph_param' in st.session_state:
            del st.session_state['graph_param']
        st.session_state.df = df
        cols = list(df.columns[:-1])
        if len(cols) == 1:
            vis = visual1D(cols, len(cols))
        elif len(cols) == 2:
            vis = visual2D(cols, len(cols))
        else:
            vis = visual(cols, len(cols))
    table = st.empty()
    table.dataframe(st.session_state.df)
    c1, c2 = st.columns([5, 1])
    c1.download_button(
        label="Export",
        data=df.to_csv().encode('utf-8'),
        file_name='data.csv',
    )
    if c2.button('Delete bottom data'):
        st.session_state.df = st.session_state.df[:-1]
        table.dataframe(st.session_state.df)

    # with st.form(key='form1'):
    col1, col2 = st.columns(2)
    batch = col1.number_input("Batch size", min_value=1,
                              help="Number of input parameter candidates suggested by BO")
    st.session_state.batch = batch
    kernel = col2.selectbox('Kernel function', ['Exponential', 'Linear', 'Matern 3/2', 'Matern 5/2',
                                                    'Radial Basis Function (RBF)', 'Rational Quadratic',
                                                    'Standard Period'],
                            index=3,
                            help="Type of kernel function used in Gaussian process regression. Matern 5/2, "
                                 "Matern 3/2, or RBF are commonly used.")
    acquisition = col1.radio("Acquisition type", options=["EI", "LCB"], index=0,
                             help="A hyperparameter of the acquisition function that determines the trade-off "
                                  "between local optimization and global search. (A large value encourages a "
                                  "global search, whereas a small value encourage local optimization)")

    ph = col2.empty()
    jitter = jwlabel(acquisition, ph)

    col1.write(" ")
    col1.write(" ")
    maximization = col1.checkbox("Maximization", help="Check this option if maximization is performed")
    noise = col1.checkbox('Noiseless', help="Check this option if noiseless evaluation is available")
    eva = col1.checkbox("Avoid re-evaluation", help="Check this option if re-evaluation for the same data is "
                                                    "avoided")
    ph_constraint = col2.empty()
    if 'value' not in st.session_state:
        st.session_state.value = ''

    cols = st.session_state.df.columns[:-1]
    with st.expander('Constraint', expanded=True):
        c9, c10, c11 = st.columns([0.7, 0.7, 4])
        with c9:
            load = st.button("Load", help='Load the range of possible values for the input parameters from data/config '
                                          'directory')
        with c10:
            save = st.button("Save", help='Save the input ranges to program execution path')
        with c11:
            saved = st.empty()
        maxmin_dic = {}
        if load:
            st.session_state.load = 1

        if 'load' not in st.session_state or st.session_state.load != 1:
            for i in cols:
                maxmin_dic[i] = [-5, 5]
            st.session_state.maxmin_dic = maxmin_dic
            record = constraint_params(maxmin_dic)
        else:
            try:
                fn = f"data/config/{uploaded_file.name.split('/')[-1].replace('.csv', '_range.csv')}"
                fn_df = pd.read_csv(fn)
            except:
                st.write("File doesn't exist. Check if the file is places at data/config directory")
            try:
                if fn_df.iloc[0,0].lower() == 'max':
                    maxx = 0
                    minn = 1
                else:
                    minn = 0
                    maxx = 1
                for i in cols:
                    maxmin_dic[i] = [fn_df[i][minn], fn_df[i][maxx]]
                st.session_state.maxmin_dic = maxmin_dic
            except:
                st.write("The loaded file have different attributes with uploaded file")
            # st.write(maxmin_dic)
            record = constraint_params(maxmin_dic)
        maxmin_dic = {}
        for i in cols:
            maxmin_dic[i] = [record[i]['Min'], record[i]['Max']]
        st.session_state.maxmin_dic = maxmin_dic
        # st.write(st.session_state.maxmin_dic)

    if save:
        columns = [''] + cols
        max_li = []
        min_li = []
        for i in record:
            max_li.append(record[i]['Max'])
            min_li.append(record[i]['Min'])
        save_df = pd.DataFrame(columns=columns, data=[max_li, min_li], index=['Max', 'Min'])
        save_df.to_csv(f"data/config/{uploaded_file.name.split('/')[-1].replace('.csv', '_range.csv')}")
        saved.write("Saved!!")
    if 'axis_record' not in st.session_state:
        st.session_state.axis_record = {}
    st.session_state.axis_record = record
    # st.text(st.session_state.axis_record)

    # update information if axis information exists
    add_constraint()
    constraint = ph_constraint.text_area("Constraints: <=0", value=st.session_state.value,
                                         help="Constraints can be defined for the input parameters by entering "
                                              "inequalities. Multiple constraints can be simultaneously defined by "
                                              "indicating multiple lines.")
    button = st.button('Run Bayesian Optimization')
    if button:
        if 'run_click' not in st.session_state:
            st.session_state.run_click = 1
        else:
            st.session_state.run_click += 1
        runBO(batch, acquisition, jitter, kernel_dic[kernel], noise, eva, maximization, constraint)
    # submit_button1 = st.form_submit_button(label='Run Bayesian Optimization')
    return table

def suggest_table(table):
    with st.form(key='form2'):
        st.subheader("Suggested data table")
        st.caption("This table is editable.")
        cols = st.session_state.df.columns
        suggest_record = []
        if 'run_record' in st.session_state:
            if len(cols) == 2:
                c1, c2 = st.columns(2)
                suggest_record = [c1.text_input(f'{cols[0]}', value=st.session_state.run_record[0][cols[0]]),
                                  c2.text_input(f'{cols[1]}', value='')]
            elif len(cols) == 3:
                c1, c2, c3 = st.columns(3)
                suggest_record = [c1.text_input(f'{cols[0]}', value=st.session_state.run_record[0][cols[0]]),
                                  c2.text_input(f'{cols[1]}', value=st.session_state.run_record[0][cols[1]]),
                                  c3.text_input(f'{cols[2]}', value='')]
            elif len(cols) == 4:
                c1, c2, c3, c4 = st.columns(4)
                suggest_record = [c1.text_input(f'{cols[0]}', value=st.session_state.run_record[0][cols[0]]),
                                  c2.text_input(f'{cols[1]}', value=st.session_state.run_record[0][cols[1]]),
                                  c3.text_input(f'{cols[2]}', value=st.session_state.run_record[0][cols[2]]),
                                  c4.text_input(f'{cols[3]}', value='')]
            else:
                c1, c2, c3, c4, c5 = st.columns(5)
                for i in range(0, len(cols), 5):
                    # st.write(st.session_state.run_record)
                    # st.write(cols)
                    suggest_record += [c1.text_input(f'{cols[i]}', value=st.session_state.run_record[0][cols[i]])]
                    if i + 1 <= len(cols)-1:
                        suggest_record += [c2.text_input(f'{cols[i+1]}', value=st.session_state.run_record[0][cols[i+1]])]
                    if i + 2 <= len(cols) - 1:
                        suggest_record += [c3.text_input(f'{cols[i+2]}', value=st.session_state.run_record[0][cols[i+2]])]
                    if i + 3 <= len(cols) - 1:
                        suggest_record += [c4.text_input(f'{cols[i+3]}', value=st.session_state.run_record[0][cols[i+3]])]
                    if i + 4 <= len(cols) - 1:
                        suggest_record += [c5.text_input(f'{cols[i+4]}', value=st.session_state.run_record[0][cols[i+4]])]
        submit_button2 = st.form_submit_button(label='Add to the imported data')
        if submit_button2:
            st.session_state.df.loc[len(st.session_state.df)] = [float(i) for i in suggest_record]
            table.dataframe(st.session_state.df)


def visualization_parameter():
    # with st.form(key='form3'):
    cols = list(st.session_state.df.columns)[:-1]
    if len(cols) == 1:
        marker_size = st.number_input("Marker size", value=10)
        make1D(marker_size)
    elif len(cols) == 2:
        cc1, cc2, cc3 = st.columns([1,1,0.5])
        res = cc1.number_input("Resolution", value=100, min_value=10, max_value=1000, step=10)
        marker_size = cc2.number_input("Marker size", value=10, min_value=1,max_value=100)
        surface = toggle.st_toggle_switch(label="3D surface",
                                      # key="Key1",
                                      default_value=False,
                                      label_after=True,
                                      inactive_color='#a1c1ae',
                                      active_color="#437F5B",
                                      track_color="#a1c1ae"
                                      )
        make2D(res, marker_size, surface)
    else:
        # st.subheader("Display axis")
        display_type_dict = {'None': 'none', 'Mean': 'mean', 'StDev': 'sd', 'Acquisiton': 'acqu'}
        default = cols[:3]
        col6, col7 = st.columns(2)
        display = st.multiselect("Display axis", cols, default=default, help="Select 3 parameters as axes of 3D graph")
        if len(display) > 3:
            st.write("Only the first 3 chosen axis will be considered")
        if len(display) >= 3:
            diss = display[:3]

        else:
            diss = default
            st.write('3 axis should be chosen!!')
        global select_axis, unselect_axis
        select_axis = sorted([i for i in range(len(cols)) if cols[i] in diss])
        unselect_axis = [i for i in range(len(cols)) if i not in select_axis]
        dis_dic = {f'{diss[0]} vs {diss[1]}': 'plane12',
                   f'{diss[0]} vs {diss[2]}': 'plane13',
                   f'{diss[1]} vs {diss[2]}': 'plane23'}
        res = col6.number_input("Resolution", value=20)
        plane = col6.radio("Display 2D plane", list(dis_dic.keys()))
        marker = col7.number_input("Marker size", value=5)
        dis = col7.radio("Display data type", ['None', 'Mean', 'StDev', 'Acquisiton'], index=1)

        # submit_button2 = st.form_submit_button(label='Make / Reload graph')
        tog = toggle.st_toggle_switch(label="Show plane",
                                      # key="Key1",
                                      default_value=False,
                                      label_after=True,
                                      inactive_color='#a1c1ae',
                                      active_color="#437F5B",
                                      track_color="#a1c1ae"
                                      )

        graph_button = st.button("Make / Reload graph")
        if graph_button:
            st.session_state.graph_param = [res, display_type_dict[dis], dis_dic[plane], display, marker]
        # try:
        if 'graph_param' in st.session_state:
            params = st.session_state.graph_param
            graph, colorbar, set_slider, slider2d = make3D(params[0], params[1], params[2], params[3], params[4])
            make3D_slider(graph, colorbar, slider2d, tog, set_slider)
        # except:
        #     st.write('Need to run Bayesian Optimization first')



def jwlabel(acquisition, ph):
    if acquisition == 'EI':
        jitter = ph.number_input('Jitter', min_value=0.00, value=0.01)
    elif acquisition == 'LCB':
        jitter = ph.number_input('Weight', min_value=0, value=2)
    return jitter


def add_constraint():
    value = ''
    if 'axis_record' in st.session_state:
        records = st.session_state.axis_record
        for i in records:
            if records[i]['Constraints']:
                value += f'{i}\n'
    st.session_state.value = value
    return value


def constraint_params(maxmin_dic):
    cols = st.session_state.df.columns[:-1]
    c3, c4, c5, c6, c7, c8 = st.columns([0.7, 0.7, 1, 1, 1, 1])
    new_title = '<p style="color:#d2e3d3;font-size: 15.8px;">1</p>'
    with c3:
        st.markdown(new_title, unsafe_allow_html=True)
        st.write("Axis name")
        axis = st.write(f'{cols[0]}')
    with c4:
        st.markdown(new_title, unsafe_allow_html=True)
        st.write("Const.")
        add = st.checkbox("Add")
    with c5:
        st.write("Range",
                 help="Set the range of possible values for the input parameters. The range can be "
                      "saved & loaded.")
        mini = st.number_input("Min", value=maxmin_dic[cols[0]][0], key='aaa')
    with c6:
        st.markdown(new_title, unsafe_allow_html=True)
        maxii = st.empty()
        maxi = maxii.number_input("Max", value=maxmin_dic[cols[0]][1], key='bbb')
    with c7:
        st.markdown(new_title, unsafe_allow_html=True)
        type = st.selectbox("Type", ['Continuous', 'Discrete'])
    with c8:
        st.markdown(new_title, unsafe_allow_html=True)
        interval = st.text_input("Interval", "1",
                                 help="Set the discretizing interval for input parameters to be suggested")
    record = {cols[0]: {'Constraints': add, 'Min': mini, 'Max': maxi, 'Type': type, 'Interval': interval}}
    new_title1 = '<p style="color:#d2e3d3;font-size: 22px;">1</p>'
    for i in range(len(cols) - 1):
        record[cols[i + 1]] = {}
        with c3:
            st.markdown(new_title1, unsafe_allow_html=True)
            st.write(cols[i + 1], key=f'{i + 1}_1')
        with c4:
            st.markdown(new_title1, unsafe_allow_html=True)
            record[cols[i + 1]]['Constraints'] = st.checkbox("Add", key=f'{i + 1}_2')
        with c5:
            record[cols[i + 1]]['Min'] = st.number_input("Min", value=st.session_state.maxmin_dic[cols[i + 1]][0],
                                                         key=f'{i + 1}_3')
        with c6:
            max_value = st.empty()
            record[cols[i + 1]]['Max'] = max_value.number_input("Max",
                                                                value=st.session_state.maxmin_dic[cols[i + 1]][1],
                                                                key=f'{i + 1}_4')
        with c7:
            record[cols[i + 1]]['Type'] = st.selectbox("Type", ['Continuous', 'Discrete'], key=f'{i + 1}_5')
        with c8:
            record[cols[i + 1]]['Interval'] = st.text_input("Interval", "1", key=f'{i + 1}_6',
                                                            help="Set the discretizing interval for input parameters to "
                                                                 "be suggested")
    return record


def runBO(batch_size, actype, jwparam, kernel, exact_fval, de_duplication, maximize, constraint):
    n_click = st.session_state.run_click
    records = st.session_state.axis_record
    type_dic = {'Continuous': 'continuous', 'Discrete': 'discrete'}
    global vmaxmin

    try:
        df = st.session_state.df

        vmaxmin = []
        cont_or_disc = []
        interval = []
        for i in records:
            value_max, value_min = records[i]['Max'], records[i]['Min']
            vmaxmin.append([value_max, value_min])
            cont_or_disc.append(type_dic[records[i]['Type']])
            interval.append(int(records[i]['Interval']))

        vis.initial_vis(df.values)
        vis.runBO_visualize(vmaxmin, actype, exact_fval, de_duplication, batch_size, jwparam, st.session_state.value.splitlines(),
                            kernel, maximize, cont_or_disc, interval)

        suggest_points = vis.BOpt.suggest_points
        results = [[''] for i in range(batch_size)]
        suggest_points = np.hstack([suggest_points, results])

        index = [str(i) for i in range(batch_size)]
        data = pd.DataFrame(data=suggest_points, index=index, columns=df.columns)

        st.write('---------------------------- Optimization Done ('+str(n_click)+' th trial) ----------------------------')

        st.session_state.run_record = data.to_dict('records')

        return 0

    except:
        st.write("Error!")
        st.session_state.run_record = None


def make1D(marker_size):
    if 'run_record' in st.session_state:
        df = st.session_state.df
        batch_size = st.session_state.batch
        graph = []
        graph_ac = []

        vis.initial_vis(df.values)
        vis.makegraph()

        graph.append(vis.setInputData(marker_size, True))
        graph.append(vis.setInputData(marker_size, False))

        graph.append(vis.setMean())
        graph.append(vis.setStDevUp())
        graph.append(vis.setStDevDown())

        graph_ac.append(vis.setAcqu())

        fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True)
        for i in range(5):
            fig.add_trace(graph[i], row=1, col=1)
        fig.add_trace(graph_ac[0], row=2, col=1)

        fig.update_xaxes(range=[vmaxmin[0][1],vmaxmin[0][0]], row=1, col=1)
        fig.update_yaxes(title='Mean and StDev', row=1, col=1)
        fig.update_xaxes(title=df.columns[0], range=[vmaxmin[0][1],vmaxmin[0][0]], row=2, col=1)
        fig.update_yaxes(title='Acquisition function', range=[0,1], row=2, col=1)
        fig.update_layout(height=800,
                          width=1000,
                          shapes = [dict(x0=vis.setSuggestData(i), x1=vis.setSuggestData(i), y0=0, y1=1,
                                         xref='x',
                                         yref='paper',
                                         line_width=2,
                                         line_color='#ff0000') for i in range(batch_size)],
                          )
        st.plotly_chart(fig)
        

def make2D(resolution, marker_size, surface):
    if 'run_record' in st.session_state:
        df = st.session_state.df
        cols = list(df.columns)

        vis.initial_vis(df.values)
        vis.makegraph(resolution)

        if not surface:
            fig_m = [vis.setInputData(marker_size, best=True, legend=True),
                     vis.setInputData(marker_size, best=False, legend=True),
                     vis.setSuggestData(marker_size, True)]
            fig_v = [vis.setInputData(marker_size, best=True, legend=False),
                     vis.setInputData(marker_size, best=False, legend=False),
                     vis.setSuggestData(marker_size, False)]
            fig_ac = [vis.setInputData(marker_size, best=True, legend=False),
                      vis.setInputData(marker_size, best=False, legend=False),
                      vis.setSuggestData(marker_size, False)]

            fig_m.append(vis.setMean())
            fig_v.append(vis.setStDev())
            fig_ac.append(vis.setAcqu())
            spec = [[{}, {}],[{}, None]]
        else:
            fig_m = [vis.setInputData3D(marker_size, best=True, legend=True),
                     vis.setInputData3D(marker_size, best=False, legend=True),
                     vis.setSuggestData3D(marker_size, True, 'mean')]
            fig_v = [vis.setSuggestData3D(marker_size, False, 'sd')]
            fig_ac = [vis.setSuggestData3D(marker_size, False, 'acqu')]

            fig_m.append(vis.setMean3D())
            fig_v.append(vis.setStDev3D())
            fig_ac.append(vis.setAcqu3D())
            spec = [[{'type': 'scene'}, {'type': 'scene'}],[{'type': 'scene'}, None]]

        fig = make_subplots(rows=2, cols=2,
                            specs=spec,
                            horizontal_spacing=0.15,
                            vertical_spacing=0.15,
                            subplot_titles=('Mean', 'StDev', 'Acquisition'))

        if not surface:
            for i in range(4):
                fig.add_trace(fig_m[i], row=1, col=1)
                fig.add_trace(fig_v[i], row=1, col=2)
                fig.add_trace(fig_ac[i], row=2, col=1)

            ratio = (vmaxmin[0][0]-vmaxmin[0][1])/(vmaxmin[1][0]-vmaxmin[1][1])

            fig.update_xaxes(title=cols[0],
                             range=[vmaxmin[0][1],vmaxmin[0][0]],
                             scaleanchor='y',
                             scaleratio=1/ratio,
                             constrain='domain',
                             constraintoward= 'right',
                             )
            fig.update_yaxes(title=cols[1],
                             range=[vmaxmin[1][1],vmaxmin[1][0]],
                             zeroline=False,
                             constrain='domain',
                             )
            fig.layout.annotations[0].update(x=0.27)
            fig.layout.annotations[1].update(x=0.85)
            fig.layout.annotations[2].update(x=0.27)

        else:
            for i in range(4):
                fig.add_trace(fig_m[i], row=1, col=1)

            for i in range(2):
                fig.add_trace(fig_v[i], row=1, col=2)
                fig.add_trace(fig_ac[i], row=2, col=1)

            camera = dict(up=dict(x=0, y=0, z=1),
                          center=dict(x=0, y=0, z=0),
                          eye=dict(x=1.25, y=-1.25, z=1.25)
                          )

            fig.update_layout(scene1 = dict(xaxis = dict(title=cols[0], range=[vmaxmin[0][1],vmaxmin[0][0]]),
                                            yaxis = dict(title=cols[1], range=[vmaxmin[1][1],vmaxmin[1][0]]),
                                            zaxis = dict(title='Mean', range=[vis.m.min(), vis.m.max()]),
                                            camera=camera,
                                            aspectmode='cube',
                                            ),
                              scene2 = dict(xaxis = dict(title=cols[0], range=[vmaxmin[0][1],vmaxmin[0][0]]),
                                            yaxis = dict(title=cols[1], range=[vmaxmin[1][1],vmaxmin[1][0]]),
                                            zaxis = dict(title='StDev', range=[vis.v.min(), vis.v.max()]),
                                            camera=camera,
                                            aspectmode='cube',
                                            ),
                              scene3 = dict(xaxis = dict(title=cols[0], range=[vmaxmin[0][1],vmaxmin[0][0]]),
                                            yaxis = dict(title=cols[1], range=[vmaxmin[1][1],vmaxmin[1][0]]),
                                            zaxis = dict(title='Acquisition', range=[0, 1]),
                                            camera=camera,
                                            aspectmode='cube',
                                            ),
                              margin=dict(r=20, l=10, b=10, t=10),
                              )

        fig.update_layout(height=800,
                          width=1000,
                          legend=dict(x=0.65, y=0.25),
                          )

        st.plotly_chart(fig)


def make3D(resolution, display_type, display_plane, axis_set, marker_size):
    """
    :param display_type: None, mean, std, acqu
    :param display_plane: plane i vs. plane j
    :param axis_set: Selected 3 parameters as axes of 3D graph in multiselect
    :return: graphspace_3D: empty space for plot
             # Output("slider_c", 'style'),
             colorbar: slider color range
             set_slider: slider unselected axis
             slider2d: slider for the axis which is not selected in "Display 2D"
             #slider_text: min, max for slider
             #text2d: min, max for slider2d
    """
    global axis_set_s, display_type_s
    axis_set_s = axis_set
    cols = st.session_state.df.columns

    graph = st.empty()

    if len(axis_set) == 3:
        df = st.session_state.df
        vis.initial_vis(df.values)

        display_type_s = display_type

        vis.setAxis(select_axis, unselect_axis)
        vis.setAxis2D(display_plane)
        vis.plot_setting(resolution, display_type, marker_size)

        vis.type_minmax()

        if display_type == 'none':
            disable = True
            range_max = 0
            range_min = 100
            vstep = 1
        else:
            disable = False
            range_max = float('{:.3g}'.format(vis.val_max))
            range_min = float('{:.3g}'.format(vis.val_min))
            vstep = (vis.val_max - vis.val_min) / 100

        # 2d slider
        slider2d = st.slider(cols[vis.unselect_axis2d[0]],
                       max_value=vmaxmin[vis.unselect_axis2d[0]][0], min_value=vmaxmin[vis.unselect_axis2d[0]][1])
        # slider2d.append(float('{:.3g}'.format(
        #     vmaxmin[vis.unselect_axis2d[0]][0])))
        # slider2d.append(float('{:.3g}'.format(
        #     vmaxmin[vis.unselect_axis2d[0]][1])))
        # slider2d.append((vmaxmin[vis.unselect_axis2d[0]]
        #                  [0] - vmaxmin[vis.unselect_axis2d[0]][1]) / 100.)

        # color slider

        colorbar = st.slider('Colorbar range', min_value=range_min, max_value=range_max, disabled=disable,
                             value=[range_min, range_max], step=vstep)

        # unselect slider
        set_slider = []
        for i in range(len(unselect_axis)):
            set_slider.append(st.slider(cols[unselect_axis[i]], key=f'slider{i}',
                                        max_value=vmaxmin[unselect_axis[i]][0], min_value=vmaxmin[unselect_axis[i]][1]))

        return graph, colorbar, set_slider, slider2d
    else:
        st.write("Error! Must choose 3 parameters.")


def make3D_slider(graph, colorbar_range, slider_value, show_plane, slider_unselect):
    """
    :param color: colorbar
    :param slider_value: slider2d
    :param show_plane: plane
    :param slider_unselect: set_slider
    :return:
    """
    axis_name = list(st.session_state.df.columns)
    dim_ex = list(slider_unselect)

    # if slicevalue > vmaxmin[vis.unselect_axis2d[0]][0]:
    #     slider_value = vmaxmin[vis.unselect_axis2d[0]][0]
    # elif slicevalue < vmaxmin[vis.unselect_axis2d[0]][1]:
    #     slider_value = vmaxmin[vis.unselect_axis2d[0]][1]
    # else:
    #     slider_value = slicevalue

    fig_point = [vis.setnullData(),
                 vis.setInputData(dim_ex, best=True),
                 vis.setInputData(dim_ex, best=False),
                 vis.setSuggestData(dim_ex)]
    fig_point2d = [vis.setnullData2D(),
                   vis.setInputData2D(slider_value, dim_ex, best=True),
                   vis.setInputData2D(slider_value, dim_ex, best=False),
                   vis.setSuggestData2D(slider_value, dim_ex)]

    vis.make3Dgraph(dim_ex)
    vis.make2Dgraph(slider_value, dim_ex)

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'scene'}, {}]],
                        horizontal_spacing=0.05)

    camera = dict(up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=2, y=-2, z=2)
                  )

    for i in range(4):
        fig.add_trace(fig_point[i], row=1, col=1)
        fig.add_trace(fig_point2d[i], row=1, col=2)

    if display_type_s == 'mean':
        fig.add_trace(vis.setMean(
            colorbar_range[0], colorbar_range[1]), row=1, col=1)
        fig.add_trace(vis.setMean2D(
            colorbar_range[0], colorbar_range[1]), row=1, col=2)
    elif display_type_s == 'sd':
        fig.add_trace(vis.setStDev(
            colorbar_range[0], colorbar_range[1]), row=1, col=1)
        fig.add_trace(vis.setStDev2D(
            colorbar_range[0], colorbar_range[1]), row=1, col=2)
    elif display_type_s == 'acqu':
        fig.add_trace(vis.setAcqu(
            colorbar_range[0], colorbar_range[1]), row=1, col=1)
        fig.add_trace(vis.setAcqu2D(
            colorbar_range[0], colorbar_range[1]), row=1, col=2)

    if show_plane:
        fig.add_trace(vis.setPlane(slider_value))

    fig.update_layout(scene=dict(xaxis=dict(title=axis_name[select_axis[0]], range=[vmaxmin[select_axis[0]][1], vmaxmin[select_axis[0]][0]],
                                            dtick=int((vmaxmin[select_axis[0]][0]-vmaxmin[select_axis[0]][1])/10)),
                                 yaxis=dict(title=axis_name[select_axis[1]], range=[vmaxmin[select_axis[1]][1], vmaxmin[select_axis[1]][0]],
                                            dtick=int((vmaxmin[select_axis[1]][0]-vmaxmin[select_axis[1]][1])/10)),
                                 zaxis=dict(title=axis_name[select_axis[2]], range=[vmaxmin[select_axis[2]][1], vmaxmin[select_axis[2]][0]],
                                            dtick=int((vmaxmin[select_axis[2]][0]-vmaxmin[select_axis[2]][1])/10))),
                      scene_aspectmode='cube',
                      scene_camera=camera,
                      margin=dict(r=20, l=10, b=10, t=10)
                      )

    ratio = (vis.X2d.max()-vis.X2d.min())/(vis.Y2d.max()-vis.Y2d.min())
    text_xaxis = axis_name[vis.select_axis2d[0]]
    text_yaxis = axis_name[vis.select_axis2d[1]]

    fig.update_xaxes(title=text_xaxis,
                     range=[vis.X2d.min(), vis.X2d.max()],
                     scaleanchor='y',
                     scaleratio=1/ratio,
                     constrain='domain',
                     constraintoward='right',
                     row=1, col=2)
    fig.update_yaxes(title=text_yaxis,
                     range=[vis.Y2d.min(), vis.Y2d.max()],
                     scaleanchor='x',
                     scaleratio=ratio,
                     zeroline=False,
                     constrain='domain',
                     row=1, col=2)

    fig.update_layout(legend=dict(x=0.),
                      scene_aspectmode='cube',
                      height=800,
                      width=1000
                      )

    graph.plotly_chart(fig)