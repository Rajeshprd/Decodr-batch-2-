# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 17:35:51 2021

@author: AJ
"""

#Hypothesis
import pandas as pd
promo = pd.read_excel("Promotion.xlsx")

promo.columns
promo.drop(['Credit card spent ($)', 'Type of promotion'],axis=1,inplace=True)

promo.columns = ['FIW',"SCO"]

#removing the null values
promo.dropna(inplace = True)


#capping the outlier
#FIW

q3 = promo.FIW.quantile(.75) 
q1 = promo.FIW.quantile(.25) 

iqr = q3 - q1

upper_extreme = q3 + (1.5*iqr)
lower_extreme = q1 - (1.5*iqr)

promo.FIW[promo.FIW > upper_extreme] = upper_extreme
promo.FIW[promo.FIW < lower_extreme]= lower_extreme


#capping for SCO
q3 = promo.SCO.quantile(.75) 
q1 = promo.SCO.quantile(.25) 

iqr = q3 - q1

upper_extreme = q3 + (1.5*iqr)
lower_extreme = q1 - (1.5*iqr)

promo.SCO[promo.SCO > upper_extreme] = upper_extreme
promo.SCO[promo.SCO < lower_extreme]= lower_extreme

#2 sample t test
from scipy import stats
stats.ttest_ind(promo.FIW,promo.SCO)

promo.describe()



""" https://pch.district70.org/pdfs/Honors-Biology-Summer-Work-part-2.pdf """


crd = pd.read_excel("ContractRenewal_Data.xlsx")

crd.columns = ['sup_A','sup_B','sup_C']

crd.isnull().sum()

crd.boxplot()

stats.shapiro(crd.sup_A)
stats.shapiro(crd.sup_B)
stats.shapiro(crd.sup_C)

#one Way ANOVA
from scipy.stats import f_oneway
f_oneway(crd.sup_A,crd.sup_B,crd.sup_C)

crd.describe()



#one sample t test
Exp_cost = pd.read_excel("ENERGY_COST.xlsx")

Exp_cost.columns = ['ID', 'cost']


from scipy.stats import ttest_1samp
ttest_1samp(Exp_cost.cost,300)

Exp_cost.cost.mean()


#normaility test

from scipy import stats 
# Ho : Data is normal
# H1 : Data is not normal

stats.shapiro(Exp_cost.cost)

#1 proportion test
from statsmodels.stats.proportion import proportions_ztest

proportions_ztest(count =482, nobs = 2000, value = 0.25)


#2 proportion test
import pandas as pd
Jhony = pd.read_excel("JohnyTalkers.xlsx")

#tabulate the data number
pd.crosstab(Jhony.Person, Jhony.Icecream)


proportions_ztest(count =[58,152], nobs = [480,740])

58/480

152/740

#chi square test
bahaman = pd.read_excel("Bahaman.xlsx")

table = pd.crosstab(bahaman.Defective, bahaman.Country)

from scipy.stats import chi2_contingency

chi2_contingency(table)



