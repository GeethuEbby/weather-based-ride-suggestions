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
