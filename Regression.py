# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 17:00:45 2021

@author: AJ
"""

#Simple Linear Regression 
import pandas as pd
wc_at = pd.read_csv("https://raw.githubusercontent.com/THEFASHIONGEEK/DATA-SCIENCE/master/Simple%20Linear%20Regression/wc.at.csv")

wc_at.isnull().sum()

wc_at.boxplot()

#no outliers and no missing values

from sklearn.model_selection import train_test_split

train, test = train_test_split(wc_at,test_size = 0.20)

# must do for simple linear regression- transpose the data
import numpy as np
X = np.array(wc_at.Waist).reshape(-1,1)
y = np.array(wc_at.AT).reshape(-1,1)

x_train , x_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LinearRegression

#to create the model
model = LinearRegression() #y = mx + C

#to train model/ fitting the model
model.fit(X = x_train,  y= y_train)

#to coeffecient of x
model.coef_

#to intercept or constant
model.intercept_

#equation of line = y = 3.5x - 221

import seaborn as sns
sns.relplot(x = "Waist",
            y = "AT",
            data = wc_at,
            kind = "scatter")

y_pred = model.predict(x_test)

# to find the accuracy of the model
model.score(X = x_test, y = y_test)


#HW : Y = bigmart sales x = Item Weight 
#score & pred values & graph scatterplot and lineplot together over one another
 