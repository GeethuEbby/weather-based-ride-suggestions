# weather-based-ride-suggestions
weather-based-ride-suggestions
## Abstract:

Modern consumers need modern solutions. This project attempts to utilize the weather forecast data to recommend vehicle for travellers when inclement weather is expected, who other wise uses alternate modes of transport. The system takes into consideration the preference of a traveller to choose pre-defined fuel-efficient vehicles among the ones available. The system not only helps customers to make smart decisions, but will also have an impact on the environment as well. 

## Problem Statement:

The objective of Mobility Recommendation – Dashboard is to generate a dashboard for a traveller where he can access weekly weather information, see any notifications arising due to inclement weather, choose vehicle recommendations and pre-book a vehicle for the day, see profile information and can access a vehicle scheduler. Vehicles are recommended for days of inclement weather and based on the passenger’s fuel preference.

## Data Flow Diagram
![Data Flow Diagram](https://github.com/GeethuEbby/weather-based-ride-suggestions/blob/d64740ba70ef928a8ebb8141fe379729e0257c8e/Data%20Flow.jpg)


The data that we have is huge and inorder to make an efficient system, we need to narrow down our region of interest. Rather than taking the whole data set into consideration, we split the data into clusters. The geo-locations data is clustered using k-means clustering algorithm. This is then merged with vehicle data and traveller data inorder to obtain the vehicles and travellers in cluster 1. We are considering Cluster 1 as our region of interest.

Now, from the weather data, we identify the days with a chance of rain. This data along with the vehicle and traveller data is passed to the recommendation engine. The engine uses haversine formula to determines the closest longitudes and latitudes to the passenger. The available vehicles are identified based on the user's preference for an electric or petrol/diesel vehicle.

The dashboard layout is built and data is passed to the dashboard. User can access the dashboard, once authenticated, and can book a suggested ride.

## Pre-Requisites:
The project is using the following version of Python interpreter.
````
	3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]
````
Install dash to bind user interface to the code. In the terminal, run the following code for installation.
````
pip install plotly
pip install plotly_express 		# To be installed for any older versions of plotly
pip install dash			# (In case of version issues, use pip install dash –upgrade. This project is using ver 2.5.1).
pip install dash-leaflet		# for interactive maps
pip install dash-extensions 		# for interactive maps
pip install dash-bootstrap-components 	# Bootstrap components for consistently styled apps with complex, responsive layouts
````
Install ipynb to allow to import ipynb modules, run the following code for installation.
````
pip install ipynb
````
## Dependent Libraries:
The following libraries need to be installed and imported as well.
````
#For data manipulation
import pandas as pd
import numpy as np
import random as r
import names
import math
import datetime
import folium

# For Dashboard
import dash
from dash import dash_table
import dash_leaflet as dl
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State
import dash_auth
import plotly.express as px
from dash.dependencies import Input, Output, State

# For importing ipynb files
import ipynb.fs
from .defs.loc_clustering import cluster_fn 	# for clustering locations based on geo code
from .defs.rain_alert_fn import rainy_days 	# function for checking inclement weather days
from .defs.prepare_map import map_html 		# function for building the map for dashboard
from .defs.vehicle_recommendation import veh_rec # function for finding the closest vehicles available for the passenger


import warnings
warnings.filterwarnings("ignore")
````
