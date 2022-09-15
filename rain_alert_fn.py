#!/usr/bin/env python
# coding: utf-8

# The rainy_days() doesnot require any inputs. When invoked, the function will read the weather dataset and will identify days in the week with any probability of rain. Inorder to display the weekly weather data on the dashboard, we also identify the 7 day range from today; and store date, average temperature of the day, day also in list. Also a list with rainy days of the week is defined. The function returns rainy day names,week days,average temperatures of the weekdays and dates of the week from today.

# In[2]:


# Importing libraries
import pandas as pd
import datetime


#Creating a column to identify chance of rain
def rain_(row):  
    if  'rain' in row['Weather']:
        return 1
    else:
        return 0



def rainy_days():
    # File contains information about week's weather - hourly data
    weather_df = pd.read_csv('WeeklyWeather.csv') 
    weather_df['Date'] =  pd.to_datetime(weather_df['Date'])
    weather_df['hour_col'] = weather_df['Date'].dt.hour
    weather_df = weather_df.apply(lambda x: x.str.lower() if x.dtype=='object' else x)

    weather_df['rain'] = weather_df.apply(lambda row: rain_(row), axis=1)
    # Identify days in the week with probability of rain
    day_delta = datetime.timedelta(days=1)
    start_date = datetime.date.today()
    end_date = start_date + 7*day_delta

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    weekly_weather = weather_df[weather_df['Date'].between(start_date, end_date)]
    #Reseting indexes to display in Weather card in dashboard
    grpWkday = weekly_weather.groupby([weekly_weather['Date'].dt.date]).mean().reset_index()
    grpWkday['Date'] = pd.to_datetime(grpWkday['Date'])
    grpWkday['Weekday'] = grpWkday['Date'].dt.day_name()
    # Identify days in the week with probability of rain
    rainy_days = []
    wkday = []
    temptre = []
    raindy = []
    wkdate = []
    for _, row in grpWkday.iterrows():
        wkday.append(row["Weekday"].capitalize())
        wkdate.append(row["Date"].date())
        temptre.append(row["Temp"])
        raindy.append(row["rain"])
        if(row["rain"] > 0):
            rainy_days.append(row["Weekday"])

    #Create a list with rainy days of the week
    rainy_days = list(dict.fromkeys(rainy_days))   
    return rainy_days,wkday,temptre,wkdate



