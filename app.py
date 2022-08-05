#!/usr/bin/env python
# coding: utf-8

# # Mobility Recommendation - Dashboard

#     Author - Geethu Susan Ebby
#     Date - 28 July 2022                            
#     Version - 0.1

# ### Objective

# Pedestrian profile is created for two users Alex and Mary with preferred travel mode = 'ebike' and fuel_preference as either electric or petrol/diesel.Vehicle profiles are also created, and driver details with contact info are added for each vehicle.Vehicles in the pedestrian cluster are identified and taxis or ubers are filtered from it. An hourly weather data is created for the week. It contains the temperature and weather type by hour. Weather Types are Clear , Cloudy, Light Rain, Low Clouds, Rain, Rain and Thunderstorm, Windy, Sunny. Based on this, the days that are expected to rain are identified, for a length of one week. 
# 
# Dashboard is created using Dash Python. 
# 
# Layout of the dashboard is as follows:
# There are three pages - Home; Notification; Profile.
# ######    * There are three pages - Home; Notification; Profile.
#     * Home tab displays the weather for the week and the location of the user.
#     * Notification tab displays any weather alerts.It shows the available vehicles that the user can choose from and allows to make booking.
#     * Profile tab displays details of the user account.
# 
# Weather is monitored for the next 7 days. If rain is expected then we try to recommend vehicle for the travellers; else notification tab shows no new notification.
# 
# 
# Vehicle recommendation is implemented as below:
# The vehicles closest to each pedestrian is identified by applying Haversine formula. Based on the pedestrian profile, the closest located vehicle with the preferred fuel type is identified. To increase the scope of finding a vehicle, three closest locations to the traveller are selected and matched to see if it it of preferred fuel type.
# 

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import random as r
import names
import math
import datetime
import folium


import dash
from dash import dash_table
import dash_leaflet as dl
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State
import dash_auth
import plotly.express as px
from dash.dependencies import Input, Output, State


import ipynb.fs
from .defs.loc_clustering import cluster_fn # for clustering locations based on geo code
from .defs.rain_alert_fn import rainy_days # function for checking inclement weather days
from .defs.prepare_map import map_html # function for building the map for dashboard
from .defs.vehicle_recommendation import veh_rec # function for finding the closest vehicles available for the passenger


import warnings
warnings.filterwarnings("ignore")


# ###### Reading Datasets

# In[2]:


# Loading csv file
pd.set_option('display.max_columns', 35)
# File contains information to geo locate the lanes and edges
edges_df = pd.read_csv('Most_edges.csv', index_col=0) 
# File contains information for all vehicles by lane, logged per hour by each minute
emission_df = pd.read_csv('most.emissionTime.csv', index_col=0) 
# File contains information about vehicle and pedestrain way points
fcdgeoTime_df = pd.read_csv('most.fcdgeoTime.csv', index_col=0)


# ###### Cleaning Dataset

# In[3]:


#Convert Time_of_Day column to a datetime format
emission_df['Time_of_Day'] =  pd.to_datetime(emission_df['Time_of_Day'])
fcdgeoTime_df['Time_of_Day'] =  pd.to_datetime(fcdgeoTime_df['Time_of_Day'])

# Split time to hours, minutes and seconds. 
# Date is bot extracted as data is recorded for '2021-08-29'
emission_df['hour_col'] = emission_df['Time_of_Day'].dt.hour
emission_df['min_col'] = emission_df['Time_of_Day'].dt.minute
emission_df['sec_col'] = emission_df['Time_of_Day'].dt.second

fcdgeoTime_df['hour_col'] = fcdgeoTime_df['Time_of_Day'].dt.hour
fcdgeoTime_df['min_col'] = fcdgeoTime_df['Time_of_Day'].dt.minute
fcdgeoTime_df['sec_col'] = fcdgeoTime_df['Time_of_Day'].dt.second

#Convert datatypes
emission_df = emission_df.astype({"timestep_time": np.int64, "vehicle_electricity": bool,
                                 "hour_col": np.int64,"min_col": np.int64,
                                 "sec_col": np.int64})
fcdgeoTime_df = fcdgeoTime_df.astype({"timestep_time": np.int64,
                                 "hour_col": np.int64,"min_col": np.int64,
                                 "sec_col": np.int64})


# rename column so as to merge dataframes, and keep column names consistent
emission_df.rename(columns = {'vehicle_lane':'laneID'}, inplace = True)
fcdgeoTime_df.rename(columns = {'vehicle_lane':'laneID'}, inplace = True)

#convert letter to lower case
emission_df = emission_df.apply(lambda x: x.str.lower() if x.dtype=='object' else x)
fcdgeoTime_df = fcdgeoTime_df.apply(lambda x: x.str.lower() if x.dtype=='object' else x)
#Removing column since there is value is only False
emission_df.drop(emission_df[emission_df.columns[pd.Series(emission_df.columns).str.startswith('vehicle_electricity')]], axis=1, inplace=True)


# In[4]:


#Clean up the column values across CSVs by removing special characters
edges_df['edgeID'] = edges_df['edgeID'].str.replace(':','')
edges_df['laneID'] = edges_df['laneID'].str.replace(':','')
edges_df['edgeID'] = edges_df['edgeID'].str.replace('#','_')
edges_df['laneID'] = edges_df['laneID'].str.replace('#','_')

emission_df['laneID'] = emission_df['laneID'].str.replace(':','')
emission_df['laneID'] = emission_df['laneID'].str.replace('#','_')
emission_df['laneID'] = emission_df['laneID'].str.replace('-','')

fcdgeoTime_df['laneID'] = fcdgeoTime_df['laneID'].str.replace(':','')
fcdgeoTime_df['laneID'] = fcdgeoTime_df['laneID'].str.replace('#','_')
fcdgeoTime_df['laneID'] = fcdgeoTime_df['laneID'].str.replace('-','')

fcdgeoTime_df['person_edge'] = fcdgeoTime_df['person_edge'].str.replace(':','')
fcdgeoTime_df['person_edge'] = fcdgeoTime_df['person_edge'].str.replace('#','_')
fcdgeoTime_df['person_edge'] = fcdgeoTime_df['person_edge'].str.replace('-','')

emission_df['vehicle_route'] = emission_df['vehicle_route'].str.replace('!italy:france','italy:france')
emission_df['vehicle_route'] = emission_df['vehicle_route'].str.replace('!var#','.')
fcdgeoTime_df['time_col'] = fcdgeoTime_df['Time_of_Day'].dt.time


# ###### Cluster the Dataset

# In[5]:


# Create a subset of edges_df so as to cluster lanes and plot on map
edges_subset_df = edges_df.loc[:,['laneID','lat','lon']]
# Using cluster function in loc_clustering notebook to allocate cluster number to geolocations
# edges_subset, column_beginning_index, column_end_index and k-value are passed to the function
# function returns dataframe with cluster value as new column
edges_subset_df = cluster_fn(edges_subset_df,1,3,7)


# In[6]:


clustered_edges_df = pd.merge(edges_df, edges_subset_df, on=['laneID', 'lat','lon'], how='outer')


# In[7]:


#Creating a sub-set dataframe -> Considering Cluster = 1
cluster1_df = clustered_edges_df[clustered_edges_df['cluster_label'].isin([1])]
# Subset without duplicates for cluster 1
cluster1_ss_df = cluster1_df.drop_duplicates(subset=['laneID'])


# In[8]:


# Subset without duplicates for cluster 1
#cluster1_df = cluster1_df.drop_duplicates(subset=['laneID'])
# Merge with all the IDs that match in both the Dataframe
cluster1_vehicle_df = pd.merge(emission_df, cluster1_ss_df, on="laneID", how="inner")


# In[9]:


#vehicle_type = 0 are Pedestrians
fcdgeoTime_df['vehicle_type'].replace(['0'],['pedestrian'], inplace=True)
#Subset dataframe filtered as pedestrian
pedestrian_fc_df = fcdgeoTime_df[fcdgeoTime_df['vehicle_type']=='pedestrian']


# In[10]:


# Remove all columns between column index 1 to 12, as they donot contain any information relevant to pedestrians
# vehicle_angle	vehicle_distance	vehicle_id	laneID	vehicle_pos	vehicle_slope	vehicle_speed	
# vehicle_type	vehicle_x	vehicle_y	vehicle_z
pedestrian_fc_df.drop(pedestrian_fc_df.iloc[:, 1:12], inplace = True, axis = 1)


# In[11]:


pedestrian_fc_df.rename(columns = {'person_edge':'edgeID'}, inplace = True)


# In[12]:


# Merge with all the IDs that match in both the Dataframe
pedest_fchr_df = pd.merge(pedestrian_fc_df, cluster1_ss_df, on="edgeID", how="inner")


# In[13]:


pedst_subset = pedest_fchr_df.groupby('person_id', as_index=False).first()
pedst_subset = pedst_subset.sort_values(by="Time_of_Day")
peds_list = pedst_subset['person_id'].tolist()
pedst_subset = pedest_fchr_df[pedest_fchr_df['person_id'].isin(peds_list[:3])]
edges_list = pedst_subset['edgeID'].unique().tolist()
edges1a_df = edges_df[edges_df['edgeID'].isin(edges_list)]
lanes_list = edges1a_df['laneID'].unique().tolist()
vehicle_cluster_df = cluster1_vehicle_df[cluster1_vehicle_df['laneID'].isin(lanes_list)]
# 'bus','hw_coach','coach','taxi','uber'
vehiclegeo_subsetdf = vehicle_cluster_df[vehicle_cluster_df['vehicle_type'].isin(['bus','hw_coach','coach','taxi','uber']) ]


# ###### Creating Profile - Traveller

# In[14]:


# Pedestrian profile is created.
pedestrian_info_df = pd.DataFrame(peds_list)
pedestrian_info_df.rename(columns = {0:'person_id'}, inplace = True)
pedestrian_preference = pedst_subset[['person_id','person_x','person_y','edgeID','laneID','cluster_label']]
pedestrian_preference = pedestrian_preference.groupby("person_id").first()
pedestrian_info_df = pedestrian_info_df.groupby("person_id").first()
pedestrian_preference = pd.merge(pedestrian_preference, pedestrian_info_df, on="person_id", how="inner")
pedestrian_preference = pedestrian_preference.head(2)
pedestrian_preference = pedestrian_preference.assign(traveller_name=["Mary Jane","Alex Joe"],
                             fuel_preference=["electric","petrol/diesel"],
                            travel_mode = ["ebike","ebike"])


# ###### Creating Profile - Vehicle

# In[15]:


#Creating the vehicle profile
ebike_travellers = pedestrian_preference.loc[pedestrian_preference['travel_mode'] == 'ebike']
edges_list = ebike_travellers['edgeID'].unique().tolist()
edges1a_df = edges_df[edges_df['edgeID'].isin(edges_list)]
lanes_list = edges1a_df['laneID'].unique().tolist()
vehicle_df = cluster1_vehicle_df[cluster1_vehicle_df['laneID'].isin(lanes_list)]

veh_ = vehicle_cluster_df[vehicle_cluster_df['vehicle_type'].isin(['taxi','uber']) ]
veh_ = veh_.groupby("vehicle_id").first()
veh_ = veh_.reset_index()
veh_ = veh_[['vehicle_id','vehicle_eclass','vehicle_fuel','laneID','vehicle_type','vehicle_x','vehicle_y','edgeID','lon','lat','cluster_label']]
fuel_type = ["electric","petrol","diesel" ]

phnum = []
count = 0
while (count < veh_.shape[0]):   
    count = count + 1
    ph_no = 902
    for i in range(1, 8):
        a = r.randint(0, 9)
        ph_no = str(ph_no) + str(a)
    phnum.append(ph_no)

driver_name = []
count = 0
while (count < veh_.shape[0]):
    count = count + 1
    driver_name.append(names.get_full_name())
    
veh_["fuel_type"] = np.random.choice(fuel_type, len(veh_))
veh_["phnum"] = np.random.choice(phnum, len(veh_))
veh_["driver_name"] = np.random.choice(driver_name, len(veh_))


# ###### Weather Info - Identifying Inclement Weather

# In[16]:


#Invoking function to identify the days with inclement weather.
# Function is defined in rain_alert_fn.ipynb
# Function returns list of days when rain is expected, in the upcoming week (ie, 7 days from today)
rainy_days,wkday,temptre,wkdate = rainy_days()


# ###### Identifying the closest vehicles

# In[17]:


# Identifiying the closest vehicles to the pedestrian
# Implemented using the haversine formula. It determines the great-circle distance between two points on a sphere given their longitudes and latitudes. 
electric_veh,gas_veh,p_points,p_name = veh_rec(ebike_travellers,veh_,rainy_days)


# In[18]:


# Subset of dataframe to be passed to dashboard
electric_veh_subset = electric_veh[['driver_name','phnum','vehicle_type','fuel_type','lat','lon']]
electric_veh = electric_veh[['driver_name','phnum','vehicle_type','fuel_type']]
electric_veh = electric_veh.rename({'driver_name': 'Driver', 'phnum': 'Phone Number', 'vehicle_type': 'Type', 'fuel_type': 'Fuel'}, axis=1)  # new method


# In[19]:


# Subset of dataframe to be passed to dashboard
gas_veh_subset = gas_veh[['driver_name','phnum','vehicle_type','fuel_type','lat','lon']]
gas_veh = gas_veh[['driver_name','phnum','vehicle_type','fuel_type']]
gas_veh = gas_veh.rename({'driver_name': 'Driver', 'phnum': 'Phone Number', 'vehicle_type': 'Type', 'fuel_type': 'Fuel'}, axis=1)  # new method


# In[20]:


lat = p_points[0][0]
lng = p_points[0][1]
fuel_type = 'electric' 

map_html(lat,lng,electric_veh_subset,fuel_type,p_name[0][0]) # Map for Mary - who prefer electric


# In[21]:


lat = p_points[1][0]
lng = p_points[1][1]
fuel_type = 'gas' 

map_html(lat,lng,electric_veh_subset,fuel_type,p_name[1][0]) # Map for Alex - who prefer Petrol/Diesel


# ### Initializing dashboard

# In[22]:


# Initializing dashboard
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=[external_stylesheets,dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)


# ##### Styling dashboard components

# In[23]:


# Dashboard style variables
colors = {
    'background': '#FFFFFF',
    'text': '#000000'
}
# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "text-align": "center"
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "1rem 1rem",
}

PAGE_STYLE = {
    "padding": "16.15rem 16.15rem",
    "-webkit-background-size": "cover",
    "background-image": "url('/assets/bkgrnd.jpg')"
}

BUTTON_STYLE = {
    "color": "white",
    "text-align": "center",
    "text-decoration": "none",
    "background-color": "rgb(13, 110, 253)",
    "padding": "1px",
    "display": "inline-block",
    'font-size':'16px',
    "height": "32px",
    "width":"65px",
    "border-radius": "9px",
}

# Dash DataTable: press on cell should highlight row
style_data_conditional = [
    {
        "if": {"state": "active"},
        "backgroundColor": "rgba(150, 180, 225, 0.2)",
        "border": "1px solid blue",
    },
    {
        "if": {"state": "selected"},
        "backgroundColor": "rgba(0, 116, 217, .03)",
        "border": "1px solid blue",
    },
]


# ###### Dashboard Layout Components

# ###### Login Page

# In[24]:


# Login Page for any user - User Name, Password & Login Button

index_page = html.Div([
html.Div(
dcc.Input(id="user", type="text", placeholder="Enter Username",className="inputbox1",
style={'margin-left':'35%','width':'450px','height':'45px','padding':'10px','margin-top':'60px',
'font-size':'16px','border-width':'3px','border-color':'#a0a3a2'
}),
),
html.Div(
dcc.Input(id="passw", type="text", placeholder="Enter Password",className="inputbox2",
style={'margin-left':'35%','width':'450px','height':'45px','padding':'10px','margin-top':'10px',
'font-size':'16px','border-width':'3px','border-color':'#a0a3a2',
}),
),
html.Div(
html.Button('Login', id='verify', n_clicks=0, style={'font-size':'16px',"color": "white","border": "none",
    "font-weight": "bold",
    "text-align": "center",
    "text-decoration": "none",
    "background-color": "rgb(13, 110, 253)",
    "display": "inline-block",
    "height": "35px",
    "border-radius": "9px"}),
style={'margin-left':'53%','padding-top':'30px'}),
html.Div(id='output1',style={'font-size':'16px',"color": "white","padding-left": "50%",
    "font-weight": "bold",
    "text-align": "center",
    "text-decoration": "none",
    "z-index": "6",
    "display": "inline-block",
    "height": "35px"})
],style=PAGE_STYLE)


# ###### Navigation Bar

# In[25]:


#Dashboard layout components - nav bar for Mary
sidebar_mary = html.Div(
[
    html.Img(
            src='/assets/image.png',   
        style={
        'vertical-align': 'middle',
        'height': '60px',
        'display': 'block',
        'margin-left': 'auto',
        'margin-right': 'auto',
        'border-radius': '50%'
    }),
    html.H2("Hello", className="display-4"),
    html.H4(p_name[0][0], className="display-6"),
    html.Hr(),
    html.P(
        "Welcome Back!", className="lead"
    ),

    dbc.Nav(
        [
            dbc.NavLink("Home", href="/next_page_1", active="exact"),
            dbc.NavLink(["Notifications ",
                        dbc.Badge(str(len(rainy_days)),color="danger",pill=True,text_color="white",className="me-1",),
], href="/notification-1", active="exact"),
            dbc.NavLink("Profile", href="/profile-1", active="exact"),
        ],
        vertical=True,
        pills=True,
    ),
],
style=SIDEBAR_STYLE,
)


# In[26]:


#Dashboard layout components - nav bar for Alex
sidebar_alex = html.Div(
[
    html.Img(
            src='/assets/image_male.png',   
        style={
        'vertical-align': 'middle',
        'height': '60px',
        'display': 'block',
        'margin-left': 'auto',
        'margin-right': 'auto',
        'border-radius': '50%'
    }),
    html.H2("Hello", className="display-4"),
    html.H4(p_name[1][0], className="display-6"),
    html.Hr(),
    html.P(
        "Welcome Back!", className="lead"
    ),

    dbc.Nav(
        [
            dbc.NavLink("Home", href="/next_page_2",  active="exact"),
            dbc.NavLink(["Notifications ",
                        dbc.Badge(str(len(rainy_days)),color="danger",pill=True,text_color="white",className="me-1",),
], href="/notification-2", active="exact"),
            dbc.NavLink("Profile", href="/profile-2", active="exact"),
        ],
        vertical=True,
        pills=True,
    ),
],
style=SIDEBAR_STYLE,
)


# ###### Logout Button

# In[27]:


# Logout Button acessible across the pages, once user logs in
logout_btn = dcc.Link('Log out', href='/',style=BUTTON_STYLE)


# ###### Weather cards - Home Page

# In[28]:


# Weather cards to be displayed on Home page, once user logs in
# Displays Day, Date anf average temperature of the day
# Weather cards are displayed for 7 days from today
weather_cards = dbc.Row(
                [
                dbc.Col(
                  dbc.Card(
                    [
                    dbc.CardImg(
                        src='/assets/wkimg.jpg',
                        top=True,
                        style={"opacity": 0.3},
                        ),
                        dbc.CardImgOverlay(
                            dbc.CardBody(
                                [
                                    html.H4(wkday[0], className="card-title"),
                                    html.P(
                                        wkdate[0],
                                        className="card-text",
                                    ),
                                    html.H4([
                                        str(round(temptre[0],2)),html.Sup(" o "),"C"],
                                        className="card-text",
                                            ),
                                ],
                            ),
                        ),
                    ],
                    style={"width": "18rem"},
                    ),
                     width={"size": 3, "order": 2},
                ),
                    dbc.Col(
                    dbc.Card(
                    [
                    dbc.CardImg(
                        src='/assets/wkimg.jpg',
                        top=True,
                        style={"opacity": 0.3},
                        ),
                        dbc.CardImgOverlay(
                            dbc.CardBody(
                                [
                                    html.H4(wkday[1], className="card-title"),
                                    html.P(
                                        wkdate[1],
                                        className="card-text",
                                    ),
                                    html.H4([
                                        str(round(temptre[1],2)),html.Sup(" o "),"C"],
                                        className="card-text",
                                            ),
                                ],
                            ),
                        ),
                    ],
                    style={"width": "18rem"},
                    ),
                         width={"size": 3, "order": 2},
                    ),
                    dbc.Col(
                    dbc.Card(
                    [
                    dbc.CardImg(
                        src='/assets/wkimg.jpg',
                        top=True,
                        style={"opacity": 0.3},
                        ),
                        dbc.CardImgOverlay(
                            dbc.CardBody(
                                [
                                    html.H4(wkday[2], className="card-title"),
                                    html.P(
                                        wkdate[2],
                                        className="card-text",
                                    ),
                                    html.H4([
                                        str(round(temptre[2],2)),html.Sup(" o "),"C"],
                                        className="card-text",
                                            ),
                                ],
                            ),
                        ),
                    ],
                    style={"width": "18rem"},
                    ),
                    width={"size": 3, "order": 2},
                    ),
                    dbc.Col(
                    dbc.Card(
                    [
                    dbc.CardImg(
                        src='/assets/wkimg.jpg',
                        top=True,
                        style={"opacity": 0.3},
                        ),
                        dbc.CardImgOverlay(
                            dbc.CardBody(
                                [
                                    html.H4(wkday[3], className="card-title"),
                                    html.P(
                                        wkdate[3],
                                        className="card-text",
                                    ),
                                    html.H4([
                                        str(round(temptre[3],2)),html.Sup(" o "),"C"],
                                        className="card-text",
                                            ),
                                ],
                            ),
                        ),
                    ],
                    style={"width": "18rem"},
                    ),
                        width={"size": 3, "order": 2},
                    ),
                    dbc.Col(
                    dbc.Card(
                    [
                    dbc.CardImg(
                        src='/assets/wkimg.jpg',
                        top=True,
                        style={"opacity": 0.3},
                        ),
                        dbc.CardImgOverlay(
                            dbc.CardBody(
                                [
                                    html.H4(wkday[4], className="card-title"),
                                    html.P(
                                        wkdate[4],
                                        className="card-text",
                                    ),
                                    html.H4([
                                        str(round(temptre[4],2)),html.Sup(" o "),"C"],
                                        className="card-text",
                                            ),
                                ],
                            ),
                        ),
                    ],
                    style={"width": "18rem"},
                    ),
                        width={"size": 3, "order": 2},
                    ),
                    dbc.Col(
                    dbc.Card(
                    [
                    dbc.CardImg(
                        src='/assets/wkimg.jpg',
                        top=True,
                        style={"opacity": 0.3},
                        ),
                        dbc.CardImgOverlay(
                            dbc.CardBody(
                                [
                                    html.H4(wkday[5], className="card-title"),
                                    html.P(
                                        wkdate[5],
                                        className="card-text",
                                    ),
                                    html.H4([
                                        str(round(temptre[5],2)),html.Sup(" o "),"C"],
                                        className="card-text",
                                            ),
                                ],
                            ),
                        ),
                    ],
                    style={"width": "18rem"},
                    ),
                        width={"size": 3, "order": 2},
                    ),
                    dbc.Col(
                    dbc.Card(
                    [
                    dbc.CardImg(
                        src='/assets/wkimg.jpg',
                        top=True,
                        style={"opacity": 0.3},
                        ),
                        dbc.CardImgOverlay(
                            dbc.CardBody(
                                [
                                    html.H4(wkday[6], className="card-title"),
                                    html.P(
                                        wkdate[6],
                                        className="card-text",
                                    ),
                                    html.H4([
                                        str(round(temptre[6],2)),html.Sup(" o "),"C"],
                                        className="card-text",
                                            ),
                                ],
                            ),
                        ),
                    ],
                    style={"width": "18rem"},
                    ),
                        width={"size": 3, "order": 2},
                    ),
                ],
                style={"flex-wrap": "nowrap","overflow-x":"scroll"},
            )


# ###### Home Page

# In[29]:


# Home Page for Mary - Prefers electric Vehicle
home_page_1 = html.Div([weather_cards,
     html.Iframe(id= 'map',srcDoc= open('maps/avail_electric_veh.html','r').read(),
                style={"height": "500px", "width": "100%"}),      
                ], style=CONTENT_STYLE) 


# Home Page for Alex - Prefers petrol/diesel Vehicle
home_page_2 = html.Div([weather_cards,
     html.Iframe(id= 'map',srcDoc= open('maps/avail_gas_veh.html','r').read(),
                style={"height": "500px", "width": "100%"}),      
                ], style=CONTENT_STYLE) 


# ###### Notification Page

# In[30]:


# Notification tab content for Mary
notification_1 = html.Div(
[
dbc.Alert(
[
    html.I(className="bi bi-exclamation-triangle-fill me-2"),
    "Upcoming Weather Alert!!!! ",
    ],
    color="danger",
    className="d-flex align-items-center",
    ),
    dbc.Row(dbc.Col(html.Div("We are expecting rain on "+rainy_days[0]+". Would you like to book a taxi for the day?"))),    
    dash_table.DataTable(
        data=electric_veh.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in gas_veh.columns],
        id='tbl',
        style_cell={'textAlign': 'left'},
        style_data_conditional=style_data_conditional,
    ),
    dbc.Alert(id='tbl_out'),
    dbc.Button("Book", id="open", n_clicks=0),
    dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Confirmation")),
        dbc.ModalBody("You have successfully booked your ride for "+rainy_days[0]+ "."),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="close", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="modal",
    is_open=False,
    ),
], style=CONTENT_STYLE) 


# ###### Notification Page

# In[31]:


# Notification tab content for Alex
notification_2 = html.Div(
[
dbc.Alert(
[
    html.I(className="bi bi-exclamation-triangle-fill me-2"),
    "Upcoming Weather Alert!!!! ",
    ],
    color="danger",
    className="d-flex align-items-center",
    ),
    dbc.Row(dbc.Col(html.Div("We are expecting rain on "+rainy_days[0]+". Would you like to book a taxi for the day?"))),    
    dash_table.DataTable(
        data=gas_veh.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in gas_veh.columns],
        id='tbl',
        style_cell={'textAlign': 'left'},
        style_data_conditional=style_data_conditional,
    ),
    dbc.Alert(id='tbl_out'),
    dbc.Button("Book", id="open", n_clicks=0),
    dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Confirmation")),
        dbc.ModalBody("You have successfully booked your ride for "+rainy_days[0]+ "."),
        dbc.ModalFooter(
            dbc.Button(
                "Close", id="close", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="modal",
    is_open=False,
    ),
], style=CONTENT_STYLE) 


# In[32]:


#When there is no Notification
no_notification_pg = html.Div(
[dbc.Alert(
    [
        html.I(className="bi bi-check-circle-fill me-2"),
        "There is no new notification!!!",
    ],
    color="success",
    style=CONTENT_STYLE,
    className="d-flex align-items-center",
),], style=CONTENT_STYLE) 
#no_notification = html.Div([sidebar, no_notification_pg])


# ###### Call Backs for Notification page

# In[33]:


# To highlight the row that is selected in the table of recommended vehicles
@app.callback(
    Output("tbl", "style_data_conditional"),
    [Input("tbl", "active_cell")]
)
def update_selected_row_color(active):
    style = style_data_conditional.copy()
    if active:
        style.append(
            {
                "if": {"row_index": active["row"]},
                "backgroundColor": "rgba(150, 180, 225, 0.2)",
                "border": "1px solid blue",
            },
        )
    return style


# In[34]:


# To show a modal confirmation box for booking conformation on Notification Page
@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


# In[35]:


# To display selected Driver on Notification Page
@app.callback(Output('tbl_out', 'children'), Input('tbl', 'active_cell'))
def update_graphs(active_cell):
    return "Click Book to confirm the ride!!! " if active_cell else "Click to select a ride!!! "


# ###### Schedule Table Layout

# In[36]:


#Scheduler table structure for Profile Page
table_header = [
    html.Thead(html.Tr([html.Th("Weekday"), html.Th("Preferred Vehicle")]))
]

row1 = html.Tr([html.Td("Sunday"), html.Td("Taxi")])
row2 = html.Tr([html.Td("Monday"), html.Td("Taxi")])
row3 = html.Tr([html.Td("Tuesday"), html.Td("Taxi")])
row4 = html.Tr([html.Td("Wednesday"), html.Td("Taxi")])
row5 = html.Tr([html.Td("Thursday"), html.Td("Taxi")])
row6 = html.Tr([html.Td("Friday"), html.Td("Taxi")])
row7 = html.Tr([html.Td("Saturday"), html.Td("Taxi")])

table_body = [html.Tbody([row1, row2, row3, row4, row5, row6, row7])]


# ###### Profile page layout

# In[37]:


# Profile page for Mary
profile_form_1 = html.Div(
                [
                dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("First Name", html_for="example-email-grid"),
                            dbc.Input(
                                id="example-email-grid",
                                value="Mary",
                                readonly=True,
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Last Name", html_for="example-password-grid"),
                            dbc.Input(
                                id="example-password-grid",
                                value="Jane",
                                readonly=True,
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="g-3",
                ),

                dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Email", html_for="example-email-grid"),
                            dbc.Input(
                                type="email",
                                id="example-email-grid",
                                value="maryjane@gmail.com",
                                readonly=True,
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Phone Number", html_for="example-password-grid"),
                            dbc.Input(

                                id="example-password-grid",
                                value="902-222-1111",
                                readonly=True,
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="g-3",
                ),
                dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Gender", html_for="example-email-grid"),
                            dbc.Input(
                                id="example-email-grid",
                                value="Female",
                                readonly=True,
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Address", html_for="example-password-grid"),
                            dbc.Input(

                                id="example-password-grid",
                                value="35 Av. Princesse Grace, 98000 Monaco",
                                readonly=True,
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="g-3",
                ),
                dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Travel Mode", html_for="example-email-grid"),
                            dbc.Input(
                                id="example-email-grid",
                                value="eBike",
                                placeholder="Enter email",
                                readonly=True,
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Fuel Preference", html_for="example-password-grid"),
                            dbc.Input(

                                id="example-password-grid",
                                value="Electric",
                                readonly=True,
                            ),
                        ],
                    ),
                ],
                className="g-3",
                ),
                dbc.Row(dbc.Col(html.H4("Scheduler"))), 
                dbc.Table(table_header + table_body, bordered=True)
                ], style=CONTENT_STYLE)


# In[38]:


# Profile page for Alex
profile_form_2 = html.Div(
                [
                dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("First Name", html_for="example-email-grid"),
                            dbc.Input(
                                id="example-email-grid",
                                value="Alex",
                                readonly=True,
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Last Name", html_for="example-password-grid"),
                            dbc.Input(
                                id="example-password-grid",
                                value="Joe",
                                readonly=True,
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="g-3",
                ),

                dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Email", html_for="example-email-grid"),
                            dbc.Input(
                                type="email",
                                id="example-email-grid",
                                value="alexjoe@gmail.com",
                                readonly=True,
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Phone Number", html_for="example-password-grid"),
                            dbc.Input(

                                id="example-password-grid",
                                value="902-222-2222",
                                readonly=True,
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="g-3",
                ),
                dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Gender", html_for="example-email-grid"),
                            dbc.Input(
                                id="example-email-grid",
                                value="Male",
                                readonly=True,
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Address", html_for="example-password-grid"),
                            dbc.Input(

                                id="example-password-grid",
                                value="1 Av. Saint-Laurent, 98000 Monaco",
                                readonly=True,
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="g-3",
                ),
                dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Travel Mode", html_for="example-email-grid"),
                            dbc.Input(
                                id="example-email-grid",
                                value="eBike",
                                placeholder="Enter email",
                                readonly=True,
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Fuel Preference", html_for="example-password-grid"),
                            dbc.Input(

                                id="example-password-grid",
                                value="Petrol/Diesel",
                                readonly=True,
                            ),
                        ],
                    ),
                ],
                className="g-3",
                ),
                dbc.Row(dbc.Col(html.H4("Scheduler"))), 
                dbc.Table(table_header + table_body, bordered=True)
                ], style=CONTENT_STYLE)


# ###### Defining Page layout

# In[39]:


#Defining the Page layouts
content = html.Div(id="page-content", style=CONTENT_STYLE)
sidebar = html.Div(id='sidebar')

app.layout = html.Div([
  dcc.Location(id='url', refresh=False),
  html.Div(id='page-content')
                     ])


# ###### Login Page and Naviagtion Call Backs

# In[40]:


# Login Page Call Back
# Authorizing User - Navigating them to appropriate login pages
# Credentials for Mary Jane - mary/mary
# Credentials for Alex Doe - alex/alex


@app.callback(
    Output('output1', 'children'), Input('verify', 'n_clicks'), State('user', 'value'), State('passw', 'value')
)
def update_output(n_clicks, uname, passw):
    li={'mary':'mary',
       'alex':'alex'}
    if uname =='' or uname == None or passw =='' or passw == None:
        return html.Div(children=' ',style={'padding-top':'10px'})
    if uname not in li:
        return html.Div(children='Incorrect Username',style={'padding-top':'40px','font-size':'16px'})
    if li[uname]==passw:
        if uname ==  "mary":
            return (dcc.Location(pathname="/next_page_1",id="someid_doesnt_matter"))
            #return html.Div([html.Div(sidebar,style={'padding-left':'550px','padding-top':'10px'}), home_page])
        else:
            return (dcc.Location(pathname="/next_page_2",id="someid_doesnt_matter"))
        #return dcc.Location(pathname="/next_page",id="someid_doesnt_matter")
        #return html.Div(dcc.Link('Access Granted!'+uname+'---'+passw, href='/next_page',style={'color':'#183d22','font-family': 'serif', 'font-weight': 'bold', "text-decoration": "none",'font-size':'20px'}),style={'padding-left':'605px','padding-top':'40px'})
    else: 
        return html.Div(children='Incorrect Password',style={'padding-top':'40px','font-size':'16px'})


# Call Back for all the pages
# Navigated based on which user has logged In
@app.callback(dash.dependencies.Output('page-content', 'children'),[dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/next_page_1':
        return html.Div([html.Div(sidebar_mary,style={'padding-left':'550px','padding-top':'10px'}),html.Div(logout_btn,
         style={'padding-left':'93%'}),home_page_1])
    elif pathname == '/next_page_2':
        return html.Div([html.Div(sidebar_alex,style={'padding-left':'550px','padding-top':'10px'}),html.Div(logout_btn,
         style={'padding-left':'93%'}), home_page_2])
    elif pathname == "/profile-1":
        return html.Div([html.Div(sidebar_mary,style={'padding-left':'550px','padding-top':'10px'}),html.Div(logout_btn,
         style={'padding-left':'93%'}), profile_form_1])
    elif pathname == "/profile-2":
        return html.Div([html.Div(sidebar_alex,style={'padding-left':'550px','padding-top':'10px'}),html.Div(logout_btn,
         style={'padding-left':'93%'}), profile_form_2])
    elif pathname == "/notification-1":
        if(len(rainy_days)>0):
            return html.Div([html.Div(sidebar_mary,style={'padding-left':'550px','padding-top':'10px'}),html.Div(logout_btn,
         style={'padding-left':'93%'}), notification_1])
        else:
            return html.Div([html.Div(sidebar_mary,style={'padding-left':'550px','padding-top':'10px'}),html.Div(logout_btn,
         style={'padding-left':'93%'}), no_notification_pg])
    elif pathname == "/notification-2":
        if(len(rainy_days)>0):
            return html.Div([html.Div(sidebar_alex,style={'padding-left':'550px','padding-top':'10px'}),html.Div(logout_btn,
         style={'padding-left':'93%'}), notification_2])
        else:
            return html.Div([html.Div(sidebar_alex,style={'padding-left':'550px','padding-top':'10px'}),html.Div(logout_btn,
         style={'padding-left':'93%'}), no_notification_pg])
    else:
        return index_page


# ### Dashboard Initialized

# In[41]:


# Initializing the dashboard.Dashboard can be accessed by clicking the URL in the output < http://127.0.0.1:8050/ >
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)


# In[ ]:




