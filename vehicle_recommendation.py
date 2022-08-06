#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import math
import numpy as np
from math import cos, asin, sqrt


# The distance() takes 2 set of latitude and logitude at a time and calculate the great-circle distance between the two points on a sphere. This is particularly used for navigation purposes.
# The closest() takes the vehicles geo-cordinates as the first input and traveller's location as the second input. Now the distance() is invoked for each vehicle location against the passenger location, and the co-ordinate with the minimum disatnce is returned as output of the closest() function.
# 
# The second_nearest() takes the vehicles geo-cordinates as the first input and traveller's location as the second input. Now the distance() is invoked for each vehicle location against the passenger location. The distances are sorted and the second minimum is returned as output of the function.
# 
# The third_nearest() takes the vehicles geo-cordinates as the first input and traveller's location as the second input. Now the distance() is invoked for each vehicle location against the passenger location. The distances are sorted and the third minimum is returned as output of the function.

# In[ ]:


# Identifiying the closest vehicles to the pedestrian
# Implemented using the haversine formula. It determines the great-circle distance between two points on a sphere given their longitudes and latitudes. 
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


# The veh_rec() function takes two inputs. The dataframe with the details of the travellers whom we are building the dashboards for, the vehicle dataset that we have identified to be considered and the list of rainy days in the week.
# The function checks if there is a rainy day in the week, if so, will proceed with excuting the followin steps.
# For each individual traveller, the latitude and longitude are identified and stored in a variable. For each vehicle traveller, the latitude and longitude are identified and stored in a variable. These variables are passed to the closest() which returns the closest geo-cordinate of the available vehicle, second_nearest() which returns the second closest geo-cordinate of the available vehicle, third_nearest() which returns the third closest geo-cordinate of the available vehicle. Now based on these locations and the fuel preference, a subset of the vehicle dataframe is identified for each passenger. The function returns the identified vehicle subsets, passenger names and geo-cordinates of passenger.

# In[15]:


def veh_rec(ebike_travellers,veh_,rainy_days):
    #rainy_days = rainy_days()
    p_points = []  
    p_name = []
    electric_veh = []
    # If rainy day is identified for the week, following code is executed to make vehicle recommendation.
    # For the pedestrian's location, the latitude and longitude are passed to the functions to identify closest,second and third nearest location.
    # Vehicle dataframe is identified that belongs to these closest locations and is of the preferred fuel type.
    if len(rainy_days) > 0:
        for grp_name, df_grp in ebike_travellers.groupby('person_id'):
            for row in df_grp.itertuples():
                if (row.traveller_name == "Mary Jane"): 
                    p_points.append([row.person_y, row.person_x])
                    p_name.append([row.traveller_name])
                    points = []
                    for vgrp_name, df_vgrp in veh_.groupby('vehicle_id'):
                        for vrow in df_vgrp.itertuples():
                            points.append([vrow.lat, vrow.lon])
                    #print(vrow)
                    closest_row = closest(points, p_points)
                    second_nearest_row = second_nearest(points, p_points)
                    third_nearest_row = third_nearest(points, p_points)
                    traveller_name = (row.traveller_name)
                    fuel_ = (row.fuel_preference)
                    electric_veh = veh_.loc[((veh_['lat'] == closest_row[0])|(veh_['lat'] == second_nearest_row[0])| (veh_['lat'] == third_nearest_row[0])) & ((veh_['fuel_type'] == 'electric') )]
                elif (row.traveller_name == "Alex Joe"): 
                    p_points.append([row.person_y, row.person_x])
                    p_name.append([row.traveller_name])
                    points = []
                    for vgrp_name, df_vgrp in veh_.groupby('vehicle_id'):
                        for vrow in df_vgrp.itertuples():
                            points.append([vrow.lat, vrow.lon])
                    #print(vrow)
                    closest_row = closest(points, p_points)
                    second_nearest_row = second_nearest(points, p_points)
                    third_nearest_row = third_nearest(points, p_points)
                    traveller_name = (row.traveller_name)
                    fuel_ = (row.fuel_preference)
                    gas_veh = veh_.loc[((veh_['lat'] == closest_row[0])|(veh_['lat'] == second_nearest_row[0])| (veh_['lat'] == third_nearest_row[0])) & ((veh_['fuel_type'] == 'diesel') | (veh_['fuel_type'] == 'petrol'))]
    return electric_veh,gas_veh,p_points,p_name

