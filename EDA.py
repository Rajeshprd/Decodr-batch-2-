# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 17:29:33 2021

@author: AJ
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #samuel norman seaborn

deliveries = pd.read_csv("https://github.com/ashutoshkrris/Data-Analysis-with-Python/raw/master/zerotopandas-course-project-starter/dataset/deliveries.csv")

matches = pd.read_csv("https://github.com/ashutoshkrris/Data-Analysis-with-Python/raw/master/zerotopandas-course-project-starter/dataset/matches.csv")


matches.isnull().sum()
matches.isnull().mean()


matches.shape

matches.Season.unique()
matches.Season.nunique()
matches.Season.value_counts()

sns.catplot(x = "Season",
            data = matches,
            kind = "count",aspect = 2)
plt.title("Matches on each year")
plt.ylabel("No of matches")


sns.catplot(x = "city",
            data = matches,
            kind = "count",aspect = 6)
plt.title("Matches hosted by cities")
plt.ylabel("No of matches")

#i want to know the details of match played on 16-04-2010
matches[matches.date == "16-04-2010"]

#i want to know the details of matches played by CSK
csk = matches[matches.team1 == 'Chennai Super Kings']
matches.team1.unique()


""" HW : Find which team won by highest margin of runs,
Find who got most player of the match, Find who won most of the matches 

and use represent in graph"""

matches[matches.win_by_runs == matches.win_by_runs.max()]


plt.figure(figsize = (8,4))
mom_player=matches.player_of_match.value_counts()[:10]

sns.barplot(x = mom_player.index, 
            y = mom_player,
            orient='v')
plt.xticks(rotation=45)


matches.player_of_match.value_counts()

sns.catplot(x = 'player_of_match', data = matches, kind = 'count')


#EDA
bigmart = pd.read_csv("Bigmart.csv")

bigmart.describe()
bigmart.info()

bigmart.shape


#check for missing values
bigmart.isnull().sum()

#Item weight and outlet type having a missing value

#############

# check for outliers
sns.boxplot(data = bigmart)

bigmart.boxplot(grid='false', color='blue',fontsize=10, rot=30)

plt.boxplot(bigmart.Item_Visibility)

#item visibility and item outlet sales are having outliers

########

#dropinng the unwated columns from the dataframe

#Item_Identifier  and Outlet_Identifier not needed 

bigmart.drop("Item_Identifier", axis = 1, inplace = True)

pd.drop

bigmart.shape
bigmart.shape[0]
bigmart.shape[1]


bigmart.drop('Outlet_Identifier',axis=1,inplace=True)

# to remove the multiple columns use []
bigmart.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)

bigmart.columns

#################
bigmart.Item_Fat_Content.value_counts()
bigmart.Item_Fat_Content.unique()
bigmart.Item_Fat_Content.nunique()

sns.catplot(x='Item_Fat_Content', kind='count',data=bigmart)

#We have only 2 fat types but represented as 5
#replace

bigmart.Item_Fat_Content = bigmart.Item_Fat_Content.str.replace('low fat',"LF") #low fat - LF

# to replace str we use function str

bigmart.Item_Fat_Content = bigmart.Item_Fat_Content.str.replace('Low Fat', "LF").replace('reg', "Regular").replace("AJ", "DS")

###########
#Item_Visibility -  probability of being seen
bigmart.Item_Visibility.median()

bigmart.Item_Visibility = bigmart.Item_Visibility.replace(0,bigmart.Item_Visibility.median())


bigmart.Item_Visibility.min()

''' use functions such replace, drop in IPL matches dataset'''

#Item_Type
bigmart.Item_Type.value_counts()
bigmart.Item_Type.unique()
bigmart.Item_Type.nunique()
#good to go

#Item_MRP
bigmart.Item_MRP.mean()
bigmart.Item_MRP.median()

plt.boxplot(bigmart.Item_MRP)
#good to go


#Outlet_Establishment_Year
bigmart.Outlet_Establishment_Year.value_counts()
#good to go

#Outlet_Size
bigmart.Outlet_Size.value_counts()

bigmart.Outlet_Size.isnull().sum()
#there is a null value


#Outlet_Location_Type
bigmart.Outlet_Location_Type.value_counts()
# good to go

#Outlet_Type
bigmart.Outlet_Type.value_counts()
# good to go

#Item_Outlet_Sales - continous data
plt.boxplot(bigmart.Item_Outlet_Sales)

#these outlier are good to go

''' Handling missing values'''

100,99,101,102,NA      

#Removing null values
100+99+101+102/4

#removing null values rows
bigmart_nona = bigmart.dropna()

# if column has more than 30% of null values remove the entire column

#to find percentage of null values
bigmart.isnull().mean()

#to the remove the entire column
bigmart.drop("Outlet_Size", axis = 1)


#imputation 
(100+99+101+102+100.5)/ 5  

#imputation by mean - if Continous data with no outlier 
bigmart.Item_Weight.fillna(bigmart.Item_Weight.mean(), inplace = True)
bigmart.Item_Weight.mean()
bigmart.isnull().sum()

#imputaion by median - if contionus data with outlier 
bigmart.Item_Weight.fillna(bigmart.Item_Weight.median(), inplace = True)
bigmart.Item_Weight.median()
bigmart.isnull().sum()


#imputation by mode
bigmart.Outlet_Size.mode()
bigmart.Outlet_Size.unique()
bigmart.Outlet_Size.value_counts()
bigmart.Outlet_Size.isnull().sum()

bigmart.Outlet_Size.fillna(bigmart.Outlet_Size.mode())

#imputaion by values
bigmart.Outlet_Size.fillna("Others",inplace = True)

"""HW : IPL dateset without null values - do needful """

''' Handling outliers in the data'''

#dropping the outlier

q3 = bigmart.Item_Outlet_Sales.quantile(.75) 
q1 = bigmart.Item_Outlet_Sales.quantile(.25) 

iqr = q3 - q1

upper_extreme = q3 + (1.5*iqr)
lower_extreme = q1 - (1.5*iqr)

bigmart_noOut = bigmart[(bigmart.Item_Outlet_Sales < upper_extreme) & (bigmart.Item_Outlet_Sales > lower_extreme )]


#caping the outlier - caping means replace outlier with upper and lower extreme

bigmart.loc[(bigmart['Item_Outlet_Sales'] > upper_extreme),bigmart['Item_Outlet_Sales']]
 = upper_extreme

bigmart.loc[(bigmart['Item_Outlet_Sales'] < lower_extreme),bigmart['Item_Outlet_Sales']]
 = lower_extreme

#EDA for the data set in the link 
# https://archive.ics.uci.edu/ml/datasets/Auto+MPG

#.data is also csv format



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

auto_data = pd.read_csv("auto-mpg.data",delim_whitespace = True,names=['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year','origin','car_name'])

#to find the null values in the data
auto_df.isnull().sum()

#horsepower has null values

sns.boxplot(data = auto_df)


#to fill the null values of horse power
auto_df.horsepower.fillna(auto_df.horsepower.median(), inplace = True)




bigmart.columns
dummy_variables = pd.get_dummies(bigmart, columns=['Item_Fat_Content','Item_Type','Outlet_Type','Outlet_Location_Type','Outlet_Size','Outlet_Establishment_Year'])


#HW : Multi linear for bigmart with dummy variables and poly transformation if needed




