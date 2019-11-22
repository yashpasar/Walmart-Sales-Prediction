import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.externals import joblib
import plotly.graph_objs as go
import dash_table
import dash_bootstrap_components as dbc
import dash_table_experiments
import base64
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.GRID])

df = joblib.load("/Users/yashpasar/Downloads/rf_test.pkl")

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

Holiday_options = []

for i in range(2):
    holiday_dict = {}
    holiday_dict['label'] = df.IsHoliday.unique()[i]
    holiday_dict['value'] = df.IsHoliday.unique()[i]
    Holiday_options.append(holiday_dict)

all_options = [
    {'label' : 'True', 'value' : 1},
    {'label' : 'False', 'value' : 0}
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


app.layout = html.Div([
    html.H1('Walmart Store Sales Prediction', style={'textAlign': 'center'}),

    dcc.Markdown(''' --- '''),

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
                  value= df['IsHoliday'].max() )], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '25%'}),

    dcc.Markdown(''' --- '''),

    html.Div([html.H3(' Weekly Sales by Department', style={'paddingRight': '30px'}),
             dcc.Graph(id='scatter-plot')]),

    html.Div([html.H3('Prediction on Test Data set using Random Forest ', style={'paddingRight': '30px'}),

    html.Div(dash_table.DataTable(
                  id='table',
                  columns=[{"name": i, "id": i} for i in df.columns],
                  fixed_rows={ 'headers': True, 'data': 0 },
                  style_table={'border': 'thin lightgrey solid'},
                  style_header={'backgroundColor':'orange','fontWeight':'bold'},
                  virtualization=True,
                  page_action='none',
                  style_as_list_view=True,
                  style_data_conditional=[{
                  'if': {'column_id': 'Weekly_Sales'},
                 'backgroundColor': '#3D9970',
                 'color': 'white' }],
                  style_cell={'width': '50px'})) ])
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
                xaxis={'title': 'Department'},
                yaxis={'title': 'Weekly Sales'},
                plot_bgcolor='rgb(230, 230,230)',
               # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                hovermode='closest')
    }

if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_hot_reload=False)