# weather-based-ride-suggestions
weather-based-ride-suggestions
## Abstract:

Modern consumers need modern solutions. This project attempts to utilize the weather forecast data to recommend vehicle for travellers when inclement weather is expected, who other wise uses alternate modes of transport. The system takes into consideration the preference of a traveller to choose pre-defined fuel-efficient vehicles among the ones available. The system not only helps customers to make smart decisions, but will also have an impact on the environment as well. 

## Problem Statement:

The objective of Mobility Recommendation – Dashboard is to generate a dashboard for a traveller where he can access weekly weather information, see any notifications arising due to inclement weather, choose vehicle recommendations and pre-book a vehicle for the day, see profile information and can access a vehicle scheduler. Vehicles are recommended for days of inclement weather and based on the passenger’s fuel preference.

## Experimental Set Up:
For development purpose, we consider traveller profiles for two users Alex and Mary, whose preferred travel mode is ebike and fuel preference is either electric or petrol/diesel vehicle. Vehicle profiles are also generated, and driver details with contact info are added for each vehicle. Vehicles in the traveller cluster are identified and taxis or Ubers are filtered from it. 

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
### Dependent Libraries:
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
## Dataset:
The project uses the following datasets in CSV format
-	most_edges.csv contains the edges/lanes ids with geo coordinates for whole edges length.
-	mostemissiontime.csv contain information for all vehicles logged per hour/minute of the day.
-	most.fcdgeo.csv contains the vehicle and pedestrian way points.
-	WeeklyWeather.csv contains the weekly weather information.


The following variables are used to store the dataset.

-	edges_df -> most_edges.csv
-	emission_df -> most.emissionTime.csv
-	fcdgeoTime_df -> most.fcdgeoTime.csv


## Data Cleaning:
It is very important to clean and format the data before proceeding any further. After using pandas library to read the csv files, the datasets are formatted as below.
-	'Time_of_Day' column in emission_df & fcdgeoTime_df data frames is converted to datetime format
-	For the above column, hours, minutes and seconds are extracted to separate columns.
-	Other columns are converted to appropriate datatype format
-	Columns are renamed to consistently access across dataframes.
-	String variables are converted to lowercase.
-	edgeID, laneID, person_edge, vehicle_route time_col columns are formatted to remove special characters.

## Clustering the Dataset
Since the dataset is rather large, we need to narrow down the area of interest. The locations available are clustered into various sections and among them one cluster is chosen for development purpose.

The clustering is done based on the geolocation. Elbow method to identify optimum number of latitude and longitude clusters. The dataset used is edges_df. A subset of the dataset is created as there are multiple value of lane and edges. Here, laneID, latitude and longitude are used for clustering.
With elbow method, k=7 is chosen and K means clustering is used to cluster. ‘cluster_label’ column contains the labelled cluster number for each row. The clustered data frame is merged with the original data frame and assigned to clustered_edges_df.

*The clustering function is defined in loc_clustering.ipynb and function name is cluster_fn()* 

![Cluster Diagram](https://github.com/GeethuEbby/weather-based-ride-suggestions/blob/ef84b0f090e248679e2caa58b293395b09c938b8/Cluster.jpg)

For all further development purpose, cluster 1 is chosen as area of interest.
The emission_df data frame (which contains the vehicle information) and fcdgeoTime_df data frame (which contains the traveller information) is merged and reassigned to new variables to based on laneId and edgeId, with clustered_edges_df. This is to get the cluster and location information for the vehicles and travellers.


## Creating Profiles
As the datasets for traveller and vehicles are ready, now we proceed to create some personalized profile information for travellers and vehicles. Note: Vehicles are filtered as taxi and uber. Since there are multiple instances for a traveller in the dataset, due to the various locations across various time period, to create the profile, we need to group the dataset based on person_id. For development purpose, two travellers are chosen and various attributes are added for them.
-	traveller_name=["Mary Jane","Alex Joe"]
-	fuel_preference=["electric","petrol/diesel"]
-	travel_mode = ["ebike","ebike"]

*The above information can be read as: Mary Jane usually travels in an ebike and prefers electric vehicles as an alternate mode of transport.*

Since there are multiple instances for a vehicle in the dataset, due to the various locations across various time period, to create the profile, we need to group the dataset based on vehicle_id. For the vehicles that are identified in cluster 1, filtered as taxi and Uber, fuel type of vehicle (electric, petrol, diesel), driver name and phone number are added to each vehicle.

## Weather Data

Our intend is to recommend alternate mode of transport for passengers in case of inclement weather. Weekly weather information which included the temperature, date, weather is read into weather_df.
To identify chance of rain, a column is added which indicate 1 if there is any chance of rain else 0. We can obtain this from the ‘Weather’ column. Now we look at the weather for the next 1 week and see if there are any days with inclement weather. Then a list is created with rainy days of the week and assigned to rainy_days.
*The clustering function is defined in rain_alert_fn.ipynb and function name is rainy_days().* 


## Vehicle Recommendation Engine and Dashboard Creation

We have gathered, cleaned and formatted all the data required for the recommendation. Now we proceed with identifying the nearest vehicles for the passenger.
*The vehicles are identified using the function defined in vehicle_recommendation.ipynb and function name is veh_rec().* 
*Vehicle data, traveller data and weather data are passed as inputs to the function.* 
````
	IF the number of rainy days in the week is greater than 1, THEN
		FOR each traveller grouped by person_id
			IF traveller_name IS Mary Jane THEN
				STORE traveller geolocation and name in variables.
				FOR each vehicle grouped by vehicle_id
					APPEND the geolocations in an array variable.
				CALL user defined function, closest WITH geo-locations of traveller and vehicles 
				RETURNS closest location to the traveller
				CALL user defined function, second_nearest WITH geo-locations of traveller and vehicles 
				RETURNS second_closest location to the traveller
				CALL user defined function, third_nearest WITH geo-locations of traveller and vehicles 
				RETURNS third_closest location to the traveller
				FILTER the vehicles with fuel type = electric and is closest, second closest and third closest to the passenger 
			IF traveller_name IS Alex Joe THEN
				STORE traveller geolocation and name in variables.
				FOR each vehicle grouped by vehicle_id
					APPEND the geolocations in an array variable.
				CALL user defined function, closest WITH geo-locations of traveller and vehicles 
				RETURNS closest location to the traveller
				CALL user defined function, second_nearest WITH geo-locations of traveller and vehicles 
				RETURNS second_closest location to the traveller
				CALL user defined function, third_nearest WITH geo-locations of traveller and vehicles 
				RETURNS third_closest location to the traveller
				FILTER the vehicles with fuel type = petrol/diesel and is closest, second closest and third closest to the passenger 
````

````
def distance(lat1, lon1, lat2, lon2 ):
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))

def closest(data, v):
    min_ = min(data, key=lambda p: distance(v[0][0],v[0][1],p[0],p[1]))
    return min_
def second_nearest(data, v):
    return sorted(data, key=lambda p: distance(v[0][0],v[0][1],p[0],p[1]))[1]
def third_nearest(data, v):
    return sorted(data, key=lambda p: distance(v[0][0],v[0][1],p[0],p[1]))[2]

````
-	The nearest points are identified by Haversine formula. It calculates great-circle distances between the two points – that is, the shortest distance over the earth’s surface, given their longitudes and latitudes.

$\ hav(c) = hav(a-b) + sin(a) sin(b) hav(C) $

-	Also, the second nearest and third nearest distances are calculated by using the same formula and sorting the results to get the second and third minimums.
-	Now variables are created with data to be passed to dashboard.
-	Map with marker location of the passenger and vehicle identified is prepared and saved as html to be displayed as iFrame in dashboard.
-	The vehicles are identified using the function defined in prepare_map.ipynb and function name is map_html().
-	Dashboard is created using dash Plotly library. 
-	External stylesheets for the dash are imported to apply styling.
-	The dashboard is initialized and dependencies are added with the below code
````
app = Dash(__name__, external_stylesheets=[external_stylesheets,dbc.themes.BOOTSTRAP, 
	dbc.icons.BOOTSTRAP, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)
````
-	Dash apps are composed of two parts. The first part is the "layout" of the app and it describes what the application looks like. The second part describes the interactivity of the application.
-	Layout elements for the dashboard are created and added to the app.
-	The layout is composed of a tree of "components" such as html.Div and dash.dcc.
-	The Dash HTML Components module (dash.html) has a component for every HTML tag.
-	The Dash Core Components module (dash.dcc) contains higher-level components that are interactive and are generated with JavaScript, HTML, and CSS through the React.js library.
-	Once the frame work is created, call back functions are added as well. The are functions that are automatically called by Dash whenever an input component's property changes, in order to update some property in another component (the output).
-	A simple authentication is added to the dashboard. User has to enter the username and password to access the dashboard.
-	Each user is navigated to their corresponding dashboard based on username. 
-	To access the dashboard, the following code is run.
````
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)

````
-	Visit http://127.0.0.1:8050/ in your web browser to view the dashboard.

## Dashboard UI Components

#### Login Page

![Login_Page](https://github.com/GeethuEbby/weather-based-ride-suggestions/blob/7c08a81115749bc68c7caf10f3ae2d822cd6940e/Login_Page.png)

- User can enter username and password to login to the dashboard. User must be authenticated to access their dashboard.
- Credentials for Mary Jane - mary/mary
- Credentials for Alex Doe - alex/alex

#### Home Page

![Home_Page](https://github.com/GeethuEbby/weather-based-ride-suggestions/blob/7c08a81115749bc68c7caf10f3ae2d822cd6940e/Home_Pg.jpg)

-	Weather cards displaying day of the week, date and expected mean weather for the day.
-	Map showing location of the logged in traveller (blue marker) and closest vehicles (red markers).

#### Notifications 

![Notification_Page](https://github.com/GeethuEbby/weather-based-ride-suggestions/blob/29b05da91777c0b4c05f236cbf93ae9fd29aad52/Notification_Pg.jpg)

-	Navigation bar shows badge if there is a notification.
-	Warning displays day when inclement weather is expected.
-	Table with details of rides that are available to choose for the logged in traveller.

#### Booking a Ride

![Booking_Page](https://github.com/GeethuEbby/weather-based-ride-suggestions/blob/29b05da91777c0b4c05f236cbf93ae9fd29aad52/Notification_Confirm_Pg.jpg)

-	Click on a row to choose a ride.
-	Click on ‘Book’ button to confirm the ride.

#### Profile

![Profile_Page](https://github.com/GeethuEbby/weather-based-ride-suggestions/blob/29b05da91777c0b4c05f236cbf93ae9fd29aad52/Profile_Pg.jpg)

-	User account information
-	User vehicle scheduler

### Use Case Diagram

##### How to Book a Ride

![UseCase](https://github.com/GeethuEbby/weather-based-ride-suggestions/blob/63e23df1b34c6d16b6f81ce33b698b77e29771fb/UseCase.jpg)

-	The traveller accesses the dashboard via URL *http://127.0.0.1:8050/*
-	User logs into the application with username and password
-	User navigates to Home Page and sees the weather forecast for the week.
-	He can also see the nearby taxis of his fuel preference.
-	User go to Notification Tab. A badge is displayed on Nav bar if there is any weather alert for the week.
-	User view the weather alert.
-	User can choose a ride from the table that is available on the page, by clicking on the desired row. 
-	User confirm the ride by clicking Book button.
-	A pop up is displayed confirming that the booking was successful.
-	User logs out of the application

### Conclusion

Based on the weather forecast, we can efficiently recommend rides for customers. By learning their preferences and travel routes, functionalities such as recommending fuel efficient alternatives, in case no recommendations of choice, can also be implemented by further improving the algorithm.




