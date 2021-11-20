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

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=123)

model = KNeighborsClassifier(5)
model.fit(X_train,y_train)

pred = model.predict(X_test)
print(confusion_matrix(y_test,pred))
model.score(X_test,y_test)

X_train.shape

import math
math.sqrt(489/2)

# max k value = no of obs of train data
# k = sqrt(n/2) best practice
# Hw create a for loop 1 to 489 for finding best model







