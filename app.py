# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 12:07:55 2021

@author: 30697
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from collections import defaultdict 
from scipy.signal import argrelextrema
from plotly.subplots import make_subplots

@st.cache
def load_data(allow_output_mutation=True):
 
    df_dict = defaultdict()
 
    df_dict['Squat Session #1'] = pd.read_csv('squats1.csv', index_col=0, header=None).T
    df_dict['Squat Session #2'] = pd.read_csv('squats2.csv', index_col=0, header=None).T
    df_dict['Squat Session #3'] = pd.read_csv('squats3.csv', index_col=0, header=None).T
    
    return df_dict 

@st.cache
def calculate_angle(df, allow_output_mutation=True):
    df['theta'] = 0
    
    for z in range(len(df)):
          
        a = np.array([float(df.iloc[z]['LEFT_HIP:x']),float(df.iloc[z]['LEFT_HIP:y'])])
        b = np.array([float(df.iloc[z]['LEFT_KNEE:x']),float(df.iloc[z]['LEFT_KNEE:y'])])
        c = np.array([float(df.iloc[z]['LEFT_HEEL:x']),float(df.iloc[z]['LEFT_HEEL:y'])])
    
        ba = b - a
        bc = b - c
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        df['theta'].iloc[z] = (np.degrees(angle))
    
    return df

st.set_page_config(page_title='JOGO AI Data Assignment',
                   page_icon='https://www.eu-startups.com/wp-content/uploads/2020/08/budbz4ey8q51oi7rehl1.png',
                   layout="wide")

_df = load_data()

# ROW 1 ------------------------------------------------------------------------

row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.beta_columns((.1, 2, 1, 1.5, .1))

row1_1.title('AI Data - Squat Analytics')

with row1_2:
    st.write('')
    row1_2.subheader('Alkiviadis Savvopoulos - Submission')
    
space, bigrow, space = st.beta_columns(
(.1, 3, .1)
)

with bigrow :
    my_opts = ['Squat Session #1', 'Squat Session #2', 'Squat Session #3']
    selected_data = st.selectbox('Select a Session', options=my_opts)  
    

# ROW 2 ------------------------------------------------------------------------

row2_spacer1, row2_1, row2_spacer3, row2_2, row2_spacer4 = st.beta_columns(
    (.1, 1.6, .1, 1.6, .1)
    )

with row2_1:

    df = calculate_angle(_df[selected_data])
    
    reps = df.iloc[argrelextrema(df.theta.values, np.less_equal,
                    order=20)[0]]['theta']
    
    reps = reps[reps<110]
    
    fig = make_subplots(
        rows=2, 
        cols=2,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}],
            [{"type": "indicator", "colspan": 2}, {"type": "indicator"}]])
    
    fig.add_trace(go.Indicator(
    mode = "number+delta",
    value = len(reps),
    title = {"text": "Attempted Repetitions"}),
                  row=1, col=1)
    
    fig.add_trace(go.Indicator(
    mode = "number+delta",
    value = 100*round(len([i for i in reps if int(i) in range(70, 90) ])/len(reps),2),
    number = {'suffix': "%"},
    title = {"text": "Healthy ROM Rep Rate"}),
                  row=1, col=2)
    
    fig.add_trace(go.Indicator(
    mode = "number+gauge+delta",
    gauge = {'axis': {'range': [50, 110]},
                 'bar': {'color': "black"},
                 'steps' : [{
                     'color': '#90ee90',
                     'line': {'color': "white", 'width': 4},
                     'thickness': 0.75,
                     'range': [70, 90]},{
                     'color': '#d3d3d3',
                     'line': {'color': 'white', 'width': 4},
                     'thickness': 0.75,
                     'range': [90, 110]},{
                     'color': 'red',
                     'line': {'color': 'white', 'width': 4},
                     'thickness': 0.75,
                     'range': [50, 70]}]

        },
    number = {'suffix': "°"},
    value = int(np.mean(reps)),
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    title = {'text': "Injury Risk Index"}),
                  row=2, col=1)
                         
    fig.update_layout(hovermode="x")

    
    st.plotly_chart(fig, use_container_width=True)
    
with row2_2:
    st.subheader('Best Repetition Range-Of-Motion')
    
    # Find best rep
    reps_minima = [int(abs(i-80)) for i in reps]
    best_rep = reps_minima.index(min(reps_minima))
    z = reps.index[best_rep]
    
    animation_frames = [go.Frame(data=[go.Scatter(x = 
                         [ 
                          float(df.iloc[z]['LEFT_HIP:y']), 
                          float(df.iloc[z]['LEFT_KNEE:y']),
                          float(df.iloc[z]['LEFT_HEEL:y'])
                          ], y = 
                         [
                          float(df.iloc[z]['LEFT_HIP:x']), 
                          float(df.iloc[z]['LEFT_KNEE:x']),
                          float(df.iloc[z]['LEFT_HEEL:x'])
                          ], mode='lines+markers+text', text=["Hip", "Knee", "Heel"], textposition="bottom center")]) for z in range(z-15, z)]
    
    z = z-16
    fig = go.Figure(
        data=[go.Scatter(x = 
                         [ 
                          float(df.iloc[z]['LEFT_HIP:y']), 
                          float(df.iloc[z]['LEFT_KNEE:y']),
                          float(df.iloc[z]['LEFT_HEEL:y'])
                          ], y = 
                         [
                          float(df.iloc[z]['LEFT_HIP:x']), 
                          float(df.iloc[z]['LEFT_KNEE:x']),
                          float(df.iloc[z]['LEFT_HEEL:x'])
                          ], mode='lines+markers+text', text=["Hip", "Knee", "Heel"], textposition="bottom center")],
        layout=go.Layout(
            xaxis=dict(range=[0.4, 0.9], autorange=False),
            yaxis=dict(range=[0, 0.8], autorange=False),
            title="Horizontal View / Left Body-side ",
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play Rep",
                              method="animate",
                              args=[None, {"frame": {"duration": 400, "redraw": True}} ])])]
        ),
        frames=animation_frames
    )
    st.plotly_chart(fig, use_container_width=True)

# ROW 3 ------------------------------------------------------------------------

row3_spacer1, row3_1, row3_spacer2 = st.beta_columns(
    (.1, 1.6, .1)
    )
    
with row3_1:
    st.subheader('Knee Angle')
        
    #fig = px.line(df['theta'], title='Hip-Heel-Knee angle through the session.')
        
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
              x=[i for i in range(len(df))]
            , y=df['theta']
            , fill='none'
            , name="Angle Progression"
        ))
        
        
    x = [i for i in range(len(df))]
    x_rev = x[::-1]
    
    y2 = [80 for i in range(len(df))]
    y2_upper =  [90 for i in range(len(df))]
    y2_lower =  [70 for i in range(len(df))]
    y2_lower = y2_lower[::-1]
    
        
    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=y2_upper+y2_lower,
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line_color='rgba(255,255,255,0)',
        name='Healthy ROM',
        showlegend=False,
    ))
    
    
    fig.add_trace(go.Scatter(
        x=x, y=y2,
        line_color='rgb(0,176,246)',
        name='Healthy ROM',
    ))
    
    fig.update_layout(title_text="Hip - heel - knee angle throughout the squatting session.")
    fig.update_layout(hovermode="x")    
    
    st.plotly_chart(fig, use_container_width=True)



