# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 16:31:19 2021

@author: AJ
"""

import scipy.stats as stats
#z-distribution
# cdf => cumulative distributive function;
stats.norm.cdf(760,711,29)  # given a value, find the probability mean =711 sd =29
stats.norm.cdf(720,711,29) 

#P(720<X<760)
stats.norm.cdf(760,711,29) - stats.norm.cdf(720,711,29)

#P(98<X<102) mean 100 sd = 2.5
stats.norm.cdf(102,100,2.5)-stats.norm.cdf(98,100,2.5)

#P(X<98)
stats.norm.cdf(98,100,2.5)

#P(X>98)
1-stats.norm.cdf(98,100,2.5)

import pandas as pd
import numpy as np

df_rain = pd.read_excel("D:/20. Decodr/Batch 2 data/Sample data.xlsx")

df_rain.columns
#Descriptive Statistics

df_rain.Rainfall.mean()
df_rain.Rainfall.median()
df_rain.Rainfall.mode()
df_rain.Rainfall.std()
df_rain.Rainfall.var()
df_rain.Rainfall.max() 
df_rain.Rainfall.min()

#range
df_rain.Rainfall.max() - df_rain.Rainfall.min()

df_rain.Rainfall.skew() 
df_rain.Rainfall.kurt()

df_rain.Rainfall.hist() 

#Q1 - First Quartile
df_rain.Rainfall.quantile(.25)
#Q3 - Third Quartile
df_rain.Rainfall.quantile(.75)


#Inter Quartile Range
df_rain.Rainfall.quantile(.75) - df_rain.Rainfall.quantile(.25)

#to describe about the dataset
df_rain.info()
df_rain.describe() #contionus data - integer columns










