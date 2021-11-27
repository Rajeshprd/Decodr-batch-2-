# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 16:29:10 2021

@author: AJ
"""

#Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)

iris.columns = ['sep_len','sep_wt','pet_len','pet_wt', 'iris_class']

iris.isnull().sum()
iris.boxplot()
iris.info()


X = iris.iloc[:,:-1]
y = iris.iris_class

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=42)

X_train.shape

#KNN - k- Nearest Neighbor
# max k value = no of obs of train data
# k = sqrt(n/2) best practice
# Hw create a for loop 1 to 489 for finding best model
model = KNeighborsClassifier(21)
model.fit(X_train,y_train)



confusion_matrix(y_test,pred)

model.score(X_test,y_test)

pred = model.predict(X_test)

actual = y_test.copy()

pd.crosstab(pred,actual)


# Decision Tree
dtmodel = DecisionTreeClassifier()
dtmodel.fit(X_train,y_train)
dtmodel.score(X_test,y_test) #testing accuracy
dtmodel.score(X_train,y_train) #training accuracy

#100% training accuracy - model is overfitted

pred = dtmodel.predict(X_test)

confusion_matrix(y_test,pred)

#displaying the DT
tree.plot_tree(dtmodel)

#ensemble learning - Together of many trees
from sklearn.ensemble import RandomForestClassifier

rfmodel = RandomForestClassifier(
    )
rfmodel.fit(X_train,y_train)
rfmodel.score(X_test,y_test) #testing accuracy
rfmodel.score(X_train,y_train) #training accuracy

from sklearn.ensemble import GradientBoostingClassifier as GB

gbmodel = GB()
gbmodel.fit(X_train,y_train)
gbmodel.score(X_test,y_test) #testing accuracy
gbmodel.score(X_train,y_train) #training accuracy

#xtreme Gredient Boosting 
from xgboost import XGBClassifier

xgmodel = XGBClassifier()
xgmodel.fit(X_train,y_train)
xgmodel.score(X_test,y_test) #testing accuracy
xgmodel.score(X_train,y_train)

help(XGBClassifier)

#adaboost - adaptive boosting

from sklearn.ensemble import AdaBoostClassifier
abmodel = AdaBoostClassifier()
abmodel.fit(X_train,y_train)
abmodel.score(X_test,y_test) #testing accuracy
abmodel.score(X_train,y_train) #training accuracy



#for loop for knn algo

for i in range(0,10):
    print(i)

result = pd.DataFrame(columns = [ "k", "score_test", "score_train"])
for k in range(1,106): 
    knnmodel = KNeighborsClassifier(k)
    knnmodel.fit(X_train,y_train)
    knnmodel.score(X_test,y_test)
    result = result.append({ "k" : k, "score_test" : knnmodel.score(X_test,y_test) , "score_train" :knnmodel.score(X_train,y_train)  },ignore_index=True)


plt.plot(result.k,result.score_test)
plt.plot(result.k,result.score_train)






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

cancer = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header= None,na_values="?")

cancer.columns   =["sample_code","thick","uni_cell_size","uni_cell_shape","mar_adh","Epi_cell_size","Bare_nuclei","bland_chro","normal_nuc","mitoses","Cancer_class"]  

cancer.shape

cancer.info() #? - na

cancer.isnull().sum() #16 na

cancer.boxplot('Bare_nuclei')
cancer.Bare_nuclei.fillna(cancer.Bare_nuclei.mean(),inplace = True)

cancer.drop('sample_code', inplace = True, axis = 1)

cancer.Cancer_class.describe()
cancer.Cancer_class.unique()

cancer.Cancer_class = cancer.Cancer_class.replace(2,"benign").replace(4,"malignant")


X = cancer.iloc[:,:-1]
y = cancer.Cancer_class

x_train,x_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=123)


#KNeighborsClassifier
knmodel = KNeighborsClassifier(18)
knmodel.fit(X = x_train, y = y_train)
knmodel.score(x_test,y_test) #0.5785714285714286
pred = knmodel.predict(x_test)
#confusion metrics
confusion_matrix(y_test, pred)


#for loop for best k value
result = pd.DataFrame(columns = [ "k", "score_test", "score_train"])
for k in range(1, 490):
    knmodel = KNeighborsClassifier(k)
    knmodel.fit(x_train,y_train)
    knmodel.score(x_test,y_test)
    result = result.append({ "k" : k, "score_test" : knmodel.score(x_test,y_test) , "score_train" :knmodel.score(x_train,y_train)  },ignore_index=True)
plt.plot(result.k,result.score_test)
plt.plot(result.k,result.score_train)

 #k value 100 - give accuracy of 
 result[result.k == 95] #93.80
 
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
Dtmodel = DecisionTreeClassifier()
Dtmodel.fit(X = x_train, y = y_train)
Dtmodel.score(x_test, y_test) #93.33
Dtmodel.score(x_train, y_train) #100
pred = Dtmodel.predict(x_test)
confusion_matrix(y_test, pred)

#displaying the decision tree
from sklearn import tree
tree.plot_tree(Dtmodel)


#ensemble learning - Together of many trees
from sklearn.ensemble import RandomForestClassifier
#random forest
rfmodel = RandomForestClassifier()
rfmodel.fit(X = x_train, y = y_train)
rfmodel.score(x_test, y_test) #97.14
rfmodel.score(x_train, y_train) #100
pred = rfmodel.predict(x_test)
confusion_matrix(y_test, pred)

from sklearn.ensemble import GradientBoostingClassifier as GB
gbmodel = GB()
gbmodel.fit(x_train,y_train)
gbmodel.score(x_test,y_test) #testing accuracy 96.19
gbmodel.score(x_train,y_train) #training accuracy #100

#xtreme Gredient Boosting 
from xgboost import XGBClassifier
xgmodel = XGBClassifier()
xgmodel.fit(x_train,y_train)
xgmodel.score(x_test,y_test) #testing accuracy
xgmodel.score(x_train,y_train) #100

#adaboost - adaptive boosting

from sklearn.ensemble import AdaBoostClassifier
abmodel = AdaBoostClassifier()
abmodel.fit(x_train,y_train)
abmodel.score(x_test,y_test) #testing accuracy 0.95
abmodel.score(x_train,y_train) #training accuracy #99.38

#best model = Random forest
final_model = RandomForestClassifier()
final_model.fit(X,y)
final_model.score(X,y) #97%


""" use according to the data """

# to create the dataframe - Create dict first with columns names of X as key
to_predict = {'thick':[8,6,4,6],'uni_cell_size':[500,390,440,414],
              'uni_cell_shape':[90,110,56,70],'mar_adh':[3555,4321,3456,2345],
              'Epi_cell_size':[10.5,11.2,5.6,7.9],'Bare_nuclei':[70,73,75,70],
              'bland_chro':[1,1,2,3],'normal_nuc' : [1,2,3,4], 'mitoses' : [1,1,1,1]}


#to convert dict to dataframe
to_predict = pd.DataFrame(to_predict)

final_model.predict(to_predict)

#flask - module - web app module 
