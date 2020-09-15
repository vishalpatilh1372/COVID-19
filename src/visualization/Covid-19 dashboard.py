import pandas as pd
import numpy as np
import dash,sys,os,subprocess
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output,State
import plotly.graph_objects as go
#import plotly.express as px

from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=True)
from scipy import signal

file_path=os.path.join(os.path.dirname(__file__),r'..\data')
sys.path.insert(0, file_path)

dir_path=os.path.join(os.path.dirname(__file__),r'..\features')
sys.path.insert(0, dir_path)

mod_path=os.path.join(os.path.dirname(__file__),r'..\models')
sys.path.insert(0, mod_path)


from import_data import getData_johns_hopkins
from relational_data import relational_dataset
from pd_large_data import result_large
from features_inventory import *




dir_path=os.path.join(os.path.dirname(__file__),r'..\..\data\raw\COVID-19')
csv_path=os.path.join(dir_path,r'..\..\processed' )
csv_path1=os.path.join(csv_path,'COVID_final_set.csv')
df_input_lg=pd.read_csv(csv_path1,sep=';')


fig = go.Figure()

app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''
    #   Enterprise Data Science - COVID 19 Data Prototype
    Task 1- Covid-19 dashboard prototype
    '''),
    
        dcc.Markdown(''' 
        ## Countries for visualization
        Choose country
        '''),
        dcc.Dropdown(
            id='country_drop_down',
            options=[ {'label': each,'value':each} for each in df_input_lg['country'].unique()],
            value=['US', 'India'], # which are pre-selected
            multi=True
        )
    ,
    
        dcc.Markdown('''
            ## Timeline of confirmed or the approximated doubling time of COVID-19 cases
            Select necessary data filter
            '''),
        dcc.Dropdown(
        id='filter_time',
        options=[
            {'label': 'Timeline Confirmed ', 'value': 'confirmed'},
            {'label': 'Timeline Confirmed Filtered', 'value': 'confirmed_filtered'},
            {'label': 'Timeline doubling Rate', 'value': 'confirmed_DR'},
            {'label': 'Timeline doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
        ],
        value='confirmed',
        multi=False),
    
    dcc.Graph(figure=fig, id='main_window_slope')
    
])



@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_drop_down', 'value'),
    Input('filter_time', 'value')])


def figure_update(country_ls,filter):


    if 'confirmed' == filter:
        my_yaxis={'type':"log",
                  'title':'Confirmed infected people (source johns hopkins csse, log-scale)'
              }
    elif 'confirmed_filtered' == filter:
        my_yaxis={'type':"log",
                  'title':'Confirmed infected people (Filtered ,log-scale)'
              }
    elif 'confirmed_DR' == filter:
        my_yaxis={'type':"log",
               'title':'Approximated doubling rate'
              }
    elif 'confirmed_filtered_DR' == filter:
        my_yaxis={'type':"log",
               'title':'Approximated doubling rate (filtered)'
              }
        

    traces = []
    for each in country_ls:

        df_plot=df_input_lg[df_input_lg['country']==each]

        if filter=='filter_rate_filtered':
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.mean).reset_index()
        else:
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.sum).reset_index()
        #print(filter)



        traces.append(dict(x=df_plot.date,
                                y=df_plot[filter],
                                mode='markers+lines',
                                opacity=8,
                                name=each
                        )
                )

    return {
            'data': traces,
            'layout': dict (
                width=2200,
                height=800,
                title= {"text": "Plot showing different trends of COVID-19 infections."},
                xaxis={'title':'Date',
                        'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#1E1E1E"),
                      },

                yaxis=my_yaxis
        )
    }

# %load ../src/features/build_features.py




def calculate_doubling_time(in_array):
    ''' Using linear regression to approximate the doubling rate

        Parameters:
        ----------
        in_array : pandas.series

        Returns:
        ----------
        doubling rate: time for doubling the number of cases
    '''

    y = np.array(in_array)
    X = np.arange(-1,2).reshape(-1, 1)

    assert len(in_array)==3
    reg.fit(X,y)
    intercept=reg.intercept_
    slope=reg.coef_

    return intercept/slope


def savgol_specs(df_input,column='confirmed',window=5):
    ''' Savgol Filter which can be used in groupby apply function (data structure kept)

        parameters:
        ----------
        df_input : pandas.series
        column : str
        window : int
            used data points to calculate the filter result

        Returns:
        ----------
        df_result: pd.DataFrame
            the index of the df_input has to be preserved in result
    '''

    degree=1
    df_result=df_input

    filter_in=df_input[column].fillna(0) # attention with the neutral element here

    result=signal.savgol_filter(np.array(filter_in),
                           window, # window size used for filtering
                           1)
    df_result[str(column+'_filtered')]=result
    return df_result

def rolling_regression(df_input,col='confirmed'):
    ''' Rolling Regression for approximating the doubling time'

        Parameters:
        ----------
        df_input: pd.DataFrame
        col: str
            defines the used column
        Returns:
        ----------
        result: pd.DataFrame
    '''
    days_back=3
    result=df_input[col].rolling(
                window=days_back,
                min_periods=days_back).apply(get_doubling_time_via_regression,raw=False)



    return result




def calculate_filtered_data(df_input,filter_on='confirmed'):
    '''  Calculate savgol filter and return merged data frame

        Parameters:
        ----------
        df_input: pd.DataFrame
        filter_on: str
            defines the used column
        Returns:
        ----------
        df_output: pd.DataFrame
            the result will be joined as a new column on the input data frame
    '''

    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'

    df_output=df_input.copy() # we need a copy here otherwise the filter_on column will be overwritten

    pd_filtered_result=df_output[['state','country',filter_on]].groupby(['state','country']).apply(savgol_filter)#.reset_index()

    #print('--+++ after group by apply')
    #print(pd_filtered_result[pd_filtered_result['country']=='Germany'].tail())

    #df_output=pd.merge(df_output,pd_filtered_result[['index',str(filter_on+'_filtered')]],on=['index'],how='left')
    df_output=pd.merge(df_output,pd_filtered_result[[str(filter_on+'_filtered')]],left_index=True,right_index=True,how='left')
    #print(df_output[df_output['country']=='Germany'].tail())
    return df_output.copy()





def calculate_filtered_data(df_input,filter_on='confirmed'):
    ''' Calculate approximated filter rate and return merged data frame

        Parameters:
        ----------
        df_input: pd.DataFrame
        filter_on: str
            defines the used column
        Returns:
        ----------
        df_output: pd.DataFrame
            the result will be joined as a new column on the input data frame
    '''

    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'


    pd_DR_result= df_input.groupby(['state','country']).apply(rolling_reg,filter_on).reset_index()

    pd_DR_result=pd_DR_result.rename(columns={filter_on:filter_on+'_DR',
                             'level_2':'index'})

    #we do the merge on the index of our big table and on the index column after groupby
    df_output=pd.merge(df_input,pd_DR_result[['index',str(filter_on+'_DR')]],left_index=True,right_on=['index'],how='left')
    df_output=df_output.drop(columns=['index'])


    return df_output



if __name__ == '__main__':

    getData_johns_hopkins()
    relational_dataset()
    result_large()
    
    print('Task-1 has ran successfully loading the server...')
    
    app.run_server(debug=True, use_reloader=False)
    