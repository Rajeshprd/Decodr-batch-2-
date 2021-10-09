# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 17:50:00 2021

@author: AJ
"""

"""dataframe - structured data - table - rows and columns """

""" pandas is a module used to handle the dataframe"""

import pandas as pd
#for importing excel file - read_excel
rainfall = pd.read_excel("D:/20. Decodr/Batch 2 data/Sample data.xlsx") #\ will not accepted in python use / or \\

org_data = rainfall.copy()    
#for importing csv file - read_csv
rainfall = pd.read_csv("D:/20. Decodr/Batch 2 data/Sample data.csv")

rainfall

#to check the column names
rainfall.columns



#to change the column names
rainfall.columns = ["mon","rain"]

#to change change column name seperatly
#change in orginal variable
rainfall.rename(columns = {"mon" : "month","rain" :"rainfall"}, inplace = True)

#create a new variable with changed name
rain = rainfall.rename(columns = {"month" : "Month"})

#to select the column
rainfall.rain
rainfall["rain"]

#to know to the number of rows and columns
rainfall.shape


#to check the first 5 rows of df
rainfall.head()

#to check the last 10 rows of df
rainfall.tail(10)

#to create column
rainfall["mon_num"] = [1,2,3,4,5,6,7,8,9,10,11,12]

#to remove a column - axis = 1
rainfall.drop("mon_num",axis = 1,inplace = True)

#to a remove a rowname 10
rainfall.drop(9,inplace = True)

#indexing in pandas - iloc - indexlocation
rainfall.iloc[6,1] #.iloc[row,column]

#slicing in pandas
rainfall.iloc[0:3,0] 
rainfall.iloc[9:12,0]

rainfall.iloc[0:3,:]# : - all the values

rainfall.iloc[:,0]

""" Task : Find mean of jan feb march rainfall """

sum(rainfall.iloc[0:3,1])/len(rainfall.iloc[0:3,1])

""" Task : Find mean of OCT NOV DEC rainfall """

#selection using the name
rainfall.loc[:,"Month"]

#selection with condition 
rainfall#[rainfall.Month == "Mar"]

rainfall[rainfall.Rainfall == 13]





