# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 16:49:00 2021

@author: AJ
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data1 = pd.read_csv("train.csv") #12 columns
data2 = pd.read_csv("test.csv") # 11 columns 

# we have to predict for the dataset named test.csv
data1.shape
data2.shape

data1.columns
data2.columns

data_y = data1['Item_Outlet_Sales']


#dropping column y in data 1
data1.drop('Item_Outlet_Sales', axis =1 , inplace = True)

data1.shape
data2.shape

#data1 is training x and data2 to predict x 

#merge two data sets
bigmart = data1.append(data2)

bigmart.columns
bigmart.drop(['Item_Identifier','Outlet_Identifier'],axis = 1, inplace = True)

#EDA
bigmart.info()
bigmart.isnull().sum()

#item Weight
bigmart.Item_Weight.mean()
bigmart.Item_Weight.fillna(bigmart.Item_Weight.mean(), inplace = True)

#'Item_Fat_Content'
bigmart['Item_Fat_Content'].unique()
bigmart.Item_Fat_Content = bigmart.Item_Fat_Content.str.replace("LF","Low_Fat").replace("low fat", "Low_Fat").replace("reg", "Regular").replace("Low Fat", "Low_Fat")

#'Item_Visibility'
bigmart.Item_Visibility.describe()
bigmart.Item_Visibility = bigmart.Item_Visibility.replace(0,bigmart.Item_Visibility.median())

#Item_Type
bigmart.Item_Type.unique() # good to go

#Item_MRP
bigmart.Item_MRP.isnull().sum() # good to go

#Outlet_Establishment_Year 
bigmart.Outlet_Establishment_Year.isnull().sum() 
bigmart.Outlet_Establishment_Year = bigmart.Outlet_Establishment_Year.astype(str)
# good to go

#Outlet_Size
bigmart.Outlet_Size.isnull().sum()
bigmart.Outlet_Size.fillna("others", inplace = True)

#Outlet_Location_Type
bigmart.Outlet_Location_Type.unique()
#good to go

bigmart.info()

#dummies
dummy = pd.get_dummies(bigmart)
dummy.shape

dummy.describe()

#unsuperivised Learning
#dimension Reduction - reduce the number of X - PCA - Principal Component Analysis







#scale - user defined function 
def scal_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x) 


normal_data = scal_func(dummy)
normal_data.describe()

#sklearn Module 
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
normal_data = scalar.fit_transform(dummy)


from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X =  normal_data)
pca_val = pd.DataFrame(pca.fit_transform(normal_data))


pca_val.shape



#variation 
var = pca.explained_variance_ratio_
var

#cum varition 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1,color="red")

#upto pca - 26 ie where most variation happens - according to the data
pca_x = pca_val.iloc[:,0:26]

pca_x.shape

#insted of dummy(41 columns) i can use pca_x(26)

#to predict the data in test.csv create a model train.csv
model_data = pca_x.iloc[0:8523,:]
to_pred = pca_x.iloc[8523:14205 ,: ]

#modelling

#splitting the data
from sklearn.model_selection import train_test_split
train_x,test_x, train_y, test_y = train_test_split(model_data,data_y, test_size = 0.2)




#linear Regression
from sklearn.linear_model import LinearRegression
#model Creation
reg_model = LinearRegression()
#model fitting
reg_model.fit(X = train_x, y = train_y)
#accuracy 
reg_model.score(test_x,test_y) # 0.53

#decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor as DTR
DTR_model = DTR()

DTR_model.fit(train_x, train_y)
DTR_model.score(test_x,test_y) #0.052



#best model is linear regression 
#prdiction
Predicted_results = reg_model.predict(to_pred)





