# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 18:12:13 2021

@author: AJ
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_rain = pd.read_excel("D:/20. Decodr/Batch 2 data/Sample data.xlsx")

#histogram - Continous data
plt.hist(df_rain.Rainfall,bins = 5,color = 'red')
#bar chart - discrete data
plt.bar(df_rain.Month,df_rain.Rainfall)


#horizontal bar chart - discrete data
plt.barh(df_rain.Month,df_rain.Rainfall)

#box plot - to find the outlier
plt.boxplot(df_rain.Rainfall,vert = False)


#scater plot - find the relation between 2 continous variables
x = [1,5,6,7,8,9,10,50,25]
y = [2,3,4,5,6,10,5,4,10]
plt.plot(x,y,".") 

#line plot
plt.plot(x,y,".--g") #fmt = [marker][line][color]
plt.plot(x,y,"v:r") 


#multiple line
x = np.array([0,2,6,8])
y1 = np.array([4,7,8,1])
y2 =np.array([6,9,10,2])
plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y1,x,y2)
plt.legend(["chennai","mumbai"])


#labels and titles
plt.plot(x,y1,x,y2)
plt.xlabel("stunumber")
plt.ylabel("height")
plt.title('height chart')
plt.grid(axis = 'x',color ='g',lw ='.5') 


#subplots

#column wise splitting
#plot1
plt.subplot(1,2,1) # 1 row and 2 columns 1 first graph
plt.plot(y1)
#plot 2
plt.subplot(1,2,2)
plt.plot(y2)

#row-wise splitting
#plot1
plt.subplot(2,1,1) # 2 row and 1 columns 1 first graph
plt.plot(y1)
#plot 2
plt.subplot(2,1,2)
plt.plot(y2)
plt.xlabel("stunumber")
plt.legend("S")

plt.show() # to diplay the plot window

''' Create a subplots with 4 (2*2) different graphs with differnt colours and labels and title

find code for changing the figure size of the graph

'''

#to add legend
plt.legend("S")

#to change the figure size
plt.figure(figsize=(6,2)) #inchs
plt.subplot(2,1,1) # 2 row and 1 columns 1 first graph
plt.plot(y1)
#plot 2
plt.subplot(2,1,2)
plt.plot(y2)
plt.xlabel("stunumber")
plt.legend("S")

#pie chart
y = np.array([35,56,79,100])
plt.pie(y, labels = [1,2,3,4])
plt.axis()

import matplotlib.pyplot as plt
import seaborn as sns #samuel norman seaborn

#to get the preloaded data set
sns.get_dataset_names()

#load the dataset from sns
tips = sns.load_dataset("tips")

tips.columns
tips.info()
tips.describe()


plt.scatter(tips.total_bill, tips.tip)
plt.xlabel("total Bill")
plt.ylabel("Tips")

#not recommended for sns - still works fine
sns.scatterplot(tips.total_bill, tips.tip)


plt.figure(figsize = (10,10))
sns.scatterplot(x = "total_bill",
                y = "tip",
                data = tips,
                hue = 'smoker')

hue_colors = { "No" : "green","Yes": "red"}

tips.smoker.unique() # to know the unique value 

plt.figure(figsize = (4,4))
sns.scatterplot(x = "total_bill",
                y = "tip",
                data = tips,
                hue = 'smoker',
                palette =  { "Yes" : "green","No": "red"},
                hue_order=["No","Yes"])

plt.title("TIPS vs Bill by Smoker/NS")
plt.xlabel("Total Bill")
plt.ylabel("Tips")

#barchart
hue_colors_mf = {"Male" : "blue", "Female" : "pink"}
sns.countplot(x= "smoker",
              data = tips,
              hue = "sex",
              palette = hue_colors_mf)

#reltional plot - continous data - scatterplot and line plot
sns.relplot(x = "total_bill",
            y = "tip",
            data = tips,
            kind = "scatter",
            hue = "smoker")

sns.relplot(x = "total_bill",
            y = "tip",
            data = tips,
            kind = "scatter",
            size = "smoker")

sns.relplot(x = "total_bill",
            y = "tip",
            data = tips,
            kind = "scatter",
            style = "smoker")

#subplotting
sns.relplot(x = "total_bill",
            y = "tip",
            data = tips,
            kind = "line",
            row = "smoker",
            col = "time")

sns.relplot(x = "total_bill",
            y = "tip",
            data = tips[0:80],
            kind = "scatter",
            style = "smoker",
            hue = 'time',
            markers = True)

""" HW : Use titanic dataset from the set of preloaded dataset, use all parameters such as hue, size, col, row, style to visualise the data PS: importantly use title labels figuresize """

#discrete data plot - catagorical plot
sns.catplot(x = "smoker",
            y = "total_bill",
            data = tips,
            kind = "bar",
            col = "day",
            col_wrap = 2)

sns.catplot(x = "day",
            y = "total_bill",
                data = tips,
                kind = "bar",ci = None)

sns.catplot(x = "day",
            y = "total_bill",
                data = tips,
                kind = "point")

#box plot
sns.catplot(x = "day",
            y = "total_bill",
                data = tips,
                kind = "box")

sns.catplot(x = "day",
            y = "total_bill",
                data = tips,
                kind = "violin")

sns.catplot(x = "day",
            y = "total_bill",
                data = tips,
                kind = "boxen")

sns.catplot(x = "day",
            y = "total_bill",
                data = tips,
                kind = "swarm")
sns.catplot(x = "day",
            y = "total_bill",
                data = tips,
                kind = "strip")


hue_colors_mf = {"Male" : "#1c4a28", "Female" : "pink"}

sns.catplot(x= "smoker",
              data = tips,
              hue = "sex",
              kind = 'count',
              palette = hue_colors_mf)

sns.catplot(x= "day",
            y= "total_bill",
            data = tips,
            kind = "violin")

sns.pairplot(tips)

sns.jointplot(x="tip", y="total_bill", data=tips, color ='green')







