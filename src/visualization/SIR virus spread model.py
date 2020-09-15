import pandas as pd
import numpy as np
import os,sys,dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State

file_path=os.path.join(os.path.dirname(__file__),r'..\data')
sys.path.insert(0, file_path)
dir_path=os.path.join(os.path.dirname(__file__),r'..\features')
sys.path.insert(0, dir_path)
mod_path=os.path.join(os.path.dirname(__file__),r'..\models')
sys.path.insert(0, mod_path)


from SIR_methods import SIR_modelling
import plotly.graph_objects as go
from scipy import optimize
from scipy import integrate

dir_path=os.path.join(os.path.dirname(__file__),r'..\..\data\raw\COVID-19')
csv_path=os.path.join(dir_path,r'..\..\processed' )
csv_path1=os.path.join(csv_path,'COVID_final_set.csv')
df_analyse = pd.read_csv(csv_path1, sep = ';')

fig = go.Figure()
app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''

    # DASH implementation for SIR model 
      Select a country for spread forecast
    '''),


    dcc.Dropdown(
        id = 'country_drop_down',
        options=[ {'label': each,'value':each} for each in df_analyse['country'].unique()],
        value= 'Germany', # which are pre-selected
        multi=False),

    dcc.Graph(figure = fig, id = 'SIR_graph')
    ])

def SIR(countries):

    SIR_modelling()


@app.callback(
    Output('SIR_graph', 'figure'),
    [Input('country_drop_down', 'value')])

def update_SIR_figure(country_drop_down):

    traces = []

    df_plot = df_analyse[df_analyse['country'] == country_drop_down]
    df_plot = df_plot[['state', 'country', 'confirmed', 'date']].groupby(['country', 'date']).agg(np.sum).reset_index()
    df_plot.sort_values('date', ascending = True).head()
    df_plot = df_plot.confirmed[35:]

    t, fitted = SIR_modelling(df_plot)

    traces.append(dict (x = t,
                        y = fitted,
                        mode = 'markers+lines',
                        opacity = 8,
                        name = 'SIR-curve')
                  )

    traces.append(dict (x = t,
                        y = df_plot,
                        mode = 'lines',
                        opacity = 0.9,
                        name = 'Original-curve')
                  )

    return {
            'data': traces,
            'layout': dict (
                width=2200,
                height=800,
                title = 'SIR virus spread model',

                xaxis= {'title':'Days',
                       'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#7f7f7f"),
                      },

                yaxis={'title': "Infected population"}
        )
    }


if __name__ == '__main__':
    print('Task-2 is running...')
    app.run_server(port=8051,debug = True, use_reloader = False)
