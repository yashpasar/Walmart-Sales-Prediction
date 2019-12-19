import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import joblib
import plotly.graph_objs as go
import dash_table
import dash_bootstrap_components as dbc
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.GRID])
server = app.server

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

# Custom Script for Heroku
if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })

df = joblib.load("rf_test.pkl")
test = joblib.load("test.pkl")
date = joblib.load("test_date.pkl")
X_train = joblib.load("X_train.pkl")
y_train = joblib.load("y_train.pkl")
X_valid = joblib.load("X_valid.pkl")
y_valid = joblib.load("y_valid.pkl")

# # # # # # # # #
external_css = ["https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
                "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "//fonts.googleapis.com/css?family=Raleway:400,300,600",
                "https://codepen.io/bcd/pen/KQrXdb.css",
                "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
                "https://codepen.io/dmcomfort/pen/JzdzEZ.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ["https://code.jquery.com/jquery-3.2.1.min.js",
               "https://codepen.io/bcd/pen/YaXojL.js"]

for js in external_js:
    app.scripts.append_script({"external_url": js})


Store_options = []

for i in range(45):
    store_dict = {}
    store_dict['label'] = df.Store.unique()[i]
    store_dict['value'] = df.Store.unique()[i]
    Store_options.append(store_dict)

Dept_options = []

for i in range(45):
    Dept_dict = {}
    Dept_dict['label'] = df.Dept.unique()[i]
    Dept_dict['value'] = df.Dept.unique()[i]
    Dept_options.append(Dept_dict)

Year_options = []

for i in range(2):
    year_dict = {}
    year_dict['label'] = df.Year.unique()[i]
    year_dict['value'] = df.Year.unique()[i]
    Year_options.append(year_dict)

Week_options = []

for i in range(39):
    week_dict = {}
    week_dict['label'] = df.Week.unique()[i]
    week_dict['value'] = df.Week.unique()[i]
    Week_options.append(week_dict)

all_options = [
    {'label' : 'True', 'value' : 1},
    {'label' : 'False', 'value' : 0}
]

intercept_options = [
    {'label' : 'True', 'value' : True},
    {'label' : 'False', 'value' : False}
]

normalize_options = [
    {'label' : 'True', 'value' : True},
    {'label' : 'False', 'value' : False}
]

copy_options = [
    {'label' : 'True', 'value' : True},
    {'label' : 'False', 'value' : False}
]

power_options = [
    {'label' : 'Manhattan Distance(L1)', 'value' : 1},
    {'label' : 'Euclidean Distance(L2)', 'value' : 2}
]

mf_options = [
    {'label' : 'auto', 'value' : 'auto'},
    {'label' : 'log2', 'value' : 'log2'},
    {'label' : 'sqrt', 'value' : 'sqrt'}
]

mapbox_access_token = 'pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w'

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(
        l=30,
        r=30,
        b=20,
        t=40
    ),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation='h'),
    title='Satellite Overview',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(
            lon=-78.05,
            lat=42.54
        ),
        zoom=7,
    )
)

app.config['suppress_callback_exceptions'] = True

app.layout = html.Div([
    html.Div([ html.H1(html.A('Sales Prediction of Walmart Stores',
                   href='https://github.com/yashpasar/Walmart-Sales-Prediction',
                   style={
                       'text-decoration': 'none',
                       'color': 'white'
                   }
                   ), style={'textAlign': 'center', 'backgroundColor':'DarkBlue'})
               ]),


    dcc.Tabs(id="tabs-example", value='tab-1-example', children=[
        dcc.Tab(label='Prediction by Best Model', value='tab-1-example'),
        dcc.Tab(label='K-Nearest Neighbors Model', value='tab-2-example'),
        dcc.Tab(label='Linear Regression Model', value='tab-3-example'),
        dcc.Tab(label='Random Forest Model', value='tab-4-example'),
        dcc.Tab(label='Support Vector Machine Model', value='tab-5-example')
    ]),

    html.Div(id='tabs-content-example')
])

@app.callback(Output('tabs-content-example', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1-example':
        return html.Div([
            html.Div([html.H3('Enter Store Number:', style={'paddingRight': '30px'}),
                      dcc.Dropdown(
                          id='my_store',
                          options=Store_options,
                          value=df['Store'].min(),
                          multi=False)], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '25%'}),

            html.Div([html.H3('Enter Week:', style={'paddingRight': '30px'}),
                      dcc.Dropdown(
                          id='my_week',
                          options=Week_options,
                          value=df['Week'].max(),
                          multi=False)], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '25%'}),

            html.Div([html.H3('Enter Year:', style={'paddingRight': '30px'}),
                      dcc.Dropdown(
                          id='my_year',
                          options=Year_options,
                          value=df['Year'].min(),
                          multi=False)], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '25%'}),

            html.Div([html.H3('Holiday:', style={'paddingRight': '30px'}),
                      dcc.RadioItems(
                          id='my_hol',
                          options=all_options,
                          value=df['IsHoliday'].max())],
                     style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '25%'}),

            dcc.Markdown(''' --- '''),

            html.Div([html.H2("Summary:", style={'paddingRight': '30px'}),
                      html.P(" We tried to solve this problem using four different algorithms out of which the Random Forest model outperformed everyone. This page provides real time interactivity in the graph and data table on using any of the filters from above. Please scroll down to see the test dataset. The other tabs allow you to run the respective model on different values of each tuning parameter and then display an important graph based on the predicted test dataset.Please wait for two or three minutes while each model produces an output. ", style={'paddingRight': '30px'})]),

            html.Div([html.H2(' Weekly Sales by Department', style={'textAlign': 'center', 'paddingRight': '30px'}),
                      dcc.Graph(id='scatter-plot')]),

            html.Div([html.H2('Prediction on Test Data set by Best Model ', style={'textAlign': 'center', 'paddingRight': '30px'}),

                      html.Div(dash_table.DataTable(
                          id='table',
                          columns=[{"name": i, "id": i} for i in df.columns],
                          fixed_rows={'headers': True, 'data': 0},
                          style_table={'border': 'thin lightgrey solid'},
                          style_header={'backgroundColor': 'orange', 'fontWeight': 'bold'},
                          virtualization=True,
                          page_action='none',
                          style_as_list_view=True,
                          style_data_conditional=[{
                              'if': {'column_id': 'Weekly_Sales'},
                              'backgroundColor': '#3D9970',
                              'color': 'white'}],
                          style_cell={'width': '50px'}))])
            ])
    elif tab == 'tab-2-example':
        return html.Div([

            html.Div([html.H4('Enter the number of neighbors:', style={'paddingRight': '30px'}),
            dcc.Slider(
                id='neighbors_slider',
                min=2,
                max=6,
                value=4,
                marks={i: i for i in [2, 3, 4, 5, 6]},
                step=1
            )], style={'width': '25%'}),

            html.Br([]),

            html.Div([html.H4('Select one of the Power Parameter:', style={'paddingRight': '30px'}),
                      dcc.RadioItems(
                          id='my_p',
                          options=power_options,
                          value=2)],
                     style={'width': '25%'}),

            html.Br([]),

            html.Div([html.H2(' Weekly Sales by Store', style={'textAlign': 'center', 'paddingRight': '30px'}),
                      dcc.Graph(id='knn-scatter-plot')])
        ])
    elif tab == 'tab-3-example':
        return html.Div([

            html.Div([html.H4('fit_intercept:', style={'paddingRight': '30px'}),
                      dcc.RadioItems(
                          id='fit_intercept',
                          options=intercept_options,
                          value=True)],
                     style={'width': '25%'}),

            html.Div([html.H4('normalize:', style={'paddingRight': '30px'}),
                      dcc.RadioItems(
                          id='normalize',
                          options=normalize_options,
                          value=False)],
                     style={'width': '25%'}),

            html.Div([html.H4('copy_X:', style={'paddingRight': '30px'}),
                      dcc.RadioItems(
                          id='copy_X',
                          options=copy_options,
                          value=True)],
                     style={'width': '25%'}),

            html.Div([html.H2(' Weekly Sales by Store', style={'textAlign': 'center', 'paddingRight': '30px'}),
                      dcc.Graph(id='lr-scatter-plot')])
        ])
    elif tab == 'tab-4-example':
        return html.Div([

            html.Div([html.H4('Select the number of trees:', style={'paddingRight': '30px'}),
            dcc.Slider(
                id='tree_slider',
                min=25,
                max=500,
                value=250,
                marks={i: i for i in [25,50,100,250,500]}
            )], style={'width': '25%'}),

            html.Br([]),

            html.Div([html.H4('Max_Features:', style={'paddingRight': '30px'}),
                      dcc.RadioItems(
                          id='my_features',
                          options=mf_options,
                          value='log2')],
                     style={'width': '25%'}),

            html.Br([]),

            html.Div([html.H2(' Weekly Sales by Store', style={'textAlign': 'center', 'paddingRight': '30px'}),
                      dcc.Graph(id='rfr-scatter-plot')])
        ])
    elif tab == 'tab-5-example':
        return html.Div([

            html.Div([html.H3('Cost(C):', style={'paddingRight': '30px'}),
            dcc.Slider(
                id='cost_slider',
                min=0.25,
                max=1,
                value=0.25,
                marks={i: i for i in [0.25, 0.5, 0.75, 1]},
                step=0.25
            )], style={'width': '25%'}),

            html.Div([html.H3('Gamma:', style={'paddingRight': '30px'}),
                      dcc.Slider(
                          id='gamma_slider',
                          min=-5,
                          max=0,
                          value=-3,
                          marks={i: '{}'.format(10 ** i) for i in
                                   range(-5, 1)},
                      )], style={'width': '25%'}),

            html.Br([]),

            html.Div([html.H2(' Weekly Sales by Store', style={'textAlign': 'center', 'paddingRight': '30px'}),
                      dcc.Graph(id='svm-scatter-plot')])
        ])


@app.callback(
    Output('table', 'data'),
    [Input('my_store', 'value'),
     Input('my_year', 'value'),
     Input('my_week', 'value'),
     Input('my_hol', 'value')])
def update_table(my_store, my_year, my_week, my_hol):
    dff = df[(df['Store'] == my_store) & (df['Year'] == my_year) & (df['Week'] == my_week) & (df['IsHoliday'] == my_hol)]
    columns = [{"name": i, "id": i} for i in dff.columns],
    data = dff.to_dict('records')
    return data

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('my_store', 'value'),
     Input('my_year', 'value'),
     Input('my_week', 'value'),
     Input('my_hol', 'value')])
def update_result(my_store, my_year, my_week, my_hol):
    dff = df[(df['Store'] == my_store) & (df['Year'] == my_year) & (df['Week'] == my_week) & (df['IsHoliday'] == my_hol)]
    return {
            'data': [
                go.Bar(
                    x=dff['Dept'],
                    y=dff['Weekly_Sales'])
                    ],
            'layout': go.Layout(
                xaxis={'title': 'Department',
                       'showline': False},
                yaxis={'title': 'Weekly Sales',
                       'showline': True,
                       'showgrid': False},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                hovermode='closest')
    }

@app.callback(
    Output('knn-scatter-plot', 'figure'),
    [Input('neighbors_slider', 'value'),
     Input('my_p', 'value')])
def update_result_knn(neighbors_slider, my_p):
    qt = QuantileTransformer(output_distribution='normal')
    knn = KNeighborsRegressor(n_neighbors=neighbors_slider, p=my_p)
    knn.fit(X_train, y_train)
    a = knn.predict(test)
    #test['Date'] = date
    return {
        'data': [
            go.Bar(
                x=test['Store'],
                y=a)
                ],
         'layout': go.Layout(
             xaxis={'title': 'Store'},
             yaxis={'title': 'Weekly Sales',
                    'showgrid': False},
             #margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
             hovermode='closest')
        }

@app.callback(
    Output('lr-scatter-plot', 'figure'),
    [Input('fit_intercept', 'value'),
     Input('normalize', 'value'),
     Input('copy_X', 'value')])
def update_result_knn(fit_intercept, normalize, copy_X):
    qt = QuantileTransformer(output_distribution='normal')
    lr = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X)
    lr.fit(X_train, y_train)
    a = lr.predict(test)
    #test['Date'] = date
    return {
        'data': [
            go.Bar(
                x=test['Store'],
                y=a)
                ],
         'layout': go.Layout(
             xaxis={'title': 'Store'},
             yaxis={'title': 'Weekly Sales'},
             # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
             hovermode='closest')
        }

@app.callback(
    Output('svm-scatter-plot', 'figure'),
    [Input('cost_slider', 'value'),
     Input('gamma_slider', 'value')])
def update_result_svm(cost_slider, gamma_slider):
    qt = QuantileTransformer(output_distribution='normal')
    clf = svm.SVR(gamma=(10**gamma_slider), C=cost_slider)
    clf.fit(X_train, y_train)
    b = clf.predict(test)
    #test['Date'] = date
    return {
        'data': [
            go.Bar(
                x=test['Store'],
                y=b)
                ],
         'layout': go.Layout(
             xaxis={'title': 'Store'},
             yaxis={'title': 'Weekly Sales'},
             # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
             hovermode='closest')
        }

@app.callback(
    Output('rfr-scatter-plot', 'figure'),
    [Input('tree_slider', 'value'),
     Input('my_features', 'value')])
def update_result_rfr(tree_slider, my_features):
    qt = QuantileTransformer(output_distribution='normal')
    rf = RandomForestRegressor(n_estimators=tree_slider, max_features=my_features)
    rf.fit(X_train, y_train)
    c = rf.predict(test)
    #test['Date'] = date
    return {
        'data': [
            go.Bar(
                x=test['Store'],
                y=c)
                ],
         'layout': go.Layout(
             xaxis={'title': 'Store'},
             yaxis={'title': 'Weekly Sales'},
             # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
             hovermode='closest')
        }


if __name__ == '__main__':
    app.run_server(debug=False)