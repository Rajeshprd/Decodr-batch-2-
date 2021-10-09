# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 17:01:44 2021

@author: AJ
"""

#Arithmetic operations
1+1 #add
2/1 #div
3*4 #mul
2-4 #sub
4/3 #div
4//3 #floor division 
11%3 #modulo
3**2 #exponential

#f9 for runing the program

#variable
160/70
height = 170 
weight = 75
BMI = height/ weight
print(BMI) #function ()

#stores the value
Height = 180 #case sensitive
#Height1 # variable name cannot be started with numbers
#no special characters in allowed expect _
#no space is allowed
height_class = 180

#naming of varible 
MyName = 'AJ' #Pascal Case
my_name = 'AJ' #snake case
myNameIsJoel = "AJ" #camel case
result = False

#data types
type(height) #integer
type(9.1) #float
type(MyName) #string
type(result) #boolen

my_height = "180 height"
my_height + my_name

result
5<4
5 == 4

print("My height is ",my_height)

#first Program 
height = int(input("Enter your height: "))
weight = int(input("Enter Weight: "))
BMI = weight/((height)**2)
print(BMI)

height = float(input("enter the height in metres: "))
weight = float(input("enter the weight in KG: "))
BMI = weight/((height)**2)
print(BMI)

height = int (input("enter your height:"))
weight = int (input("enter height:"))
bmi = height *2 / weight
print(bmi)

#HW 
'''
Create a program for finding average marks of your 10th class '''

# set of values - array of values 
#list - []
#mutable - changable
ht_class = [180,170,168,150]
ht_class[0]
ht_class[0] = 200 
ht_class[0]
#tuple ()
#immutable - cannot be changed
ht_class_tuple = (180,170,168,150)
ht_class_tuple[0] = 2000
print(ht_class_tuple)
ht_class_tuple
# set {} - unordered
ht_class_set = {180,170,168,150}

#dict { key : values } - mutable
ht_class_dict = {"stu1" : 180, 
                 "stu2": 170,
                 "stu3" : 168,
                 "stu4" : 150}
type(ht_class)
type(ht_class_tuple)
type(ht_class_set)
type(ht_class_dict)

''' HW 1 :Find difference between all 4 varibale types '''

ht_class
sum(ht_class)
len(ht_class) #len funciton give no of items
help(len) 

avg_clas = sum(ht_class)/len(ht_class)


ht_class = [180,180,135,158,100]
ht_class
sum(ht_class)

#indexing
#strats from 0 
ht_class[3]

#slicing
ht_class[0:3] #start - inclusive : end - exclusive
ht_class[2:7]

#reverse indexing
ht_class[-2]

#revere slicing 
ht_class[-3:-1]

#step Slicing
ht_class[1:5:1] #[start:end: step]





'''HW 2: create list with 20 values and find average of middle 10 numbers'''

import numpy as np #numicarl python 
np.mean(ht_class)

ht_array = np.array([180,158,170,168,170])
type(ht_array)

#to know the data type of array
ht_array.dtype

#to know the shape of array
ht_array.shape

ht_array.ndim

'''HW 3: do the above HW2 with the help numpy '''

#2D array
b = np.array([[4,7,9,11,13],[400,500,600,700,800],[400,500,600,700,800]])

b
b.shape
len(b)

b.ndim

b[0][0]
b[0,0]

c= np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
c
c.ndim
c.shape

c[0,1,0]
c[1,0,1:3]
c[1,0,1:3]


''' HW : create and index 2D and 3D(multi dim array) array '''

a = np.array([10,20,30,40,50,60,70,80,90,100,110,120])
a.shape
a.ndim

#reshaping
d = a.reshape(2,2,3)
d.ndim

#searching a value in array
np.where(c == 7)


#dict will not allow duplictes in keys
car = { "name" : "Hyundai",
       "model" : "i20",
       "engine_capacity" : 1.2,
       "seating_capacity" : 4,
       "power_windows": True} 

car["engine_capacity"]

#for copying variable
car1 = car.copy() #best practice
car2 = dict(car)

#create variable by reference
car3 = car #call by reference - not a copy 

len(car)

#for the extracting the data
car["engine_capacity"]
car.get("model")

#to get the keys in dict
car.keys()

#to get the values of the dict
car.values()

#all the items of dict
car.items()

#adding values in dict
car["colur"] = "Red"
car.update({"doors": 4})

car
car3

''' Conditional Statements ''' # - single comment ''' - multi line comments
 
age = 17

if age < 18 : #condition 
    print("He is a minor") #action - function 
    

""" conditions
a < b  - less than
a > b - greater than
a <= b -less or equal to
a >= b - greater than or equal to 
a != b - not equal to
a == b - equal 

= - is assignment operator
== - is comparision operator
"""


a = 5 
b = 5

a == b

#if and else for one condtion true and false
age = int(input("enter age: "))
if age < 18 : #condition is True
    print("You are a Minor")
else: #condition is false
    print("Your are a Major")

colour = input("enter your fav colour : ")
if colour == "yellow" :
    print("You're a CSK fan")
else :
    print("you're not CSK fan")    


#mulitple conditions use elif
colour = input("enter your fav colour : ")
if colour == "yellow" :
    print("You're a CSK fan")
elif colour == "Blue" :
    print("You're a MI fan")
else :
    print("you're a RCB fan")   

#use or, and for giving muliple conditional statments in the same line
colour = input("enter your fav colour : ")
if colour == "yellow" or colour == "Yellow" :
    print("You're a CSK fan")
elif colour == "Blue" :
    print("You're a MI fan")
else :
    print("you're a RCB fan")


age = int(input("enter age: "))
if age > 13 and age < 19: 
    print("You are a teenager")
else: #condition is false
    print("Your are not a teenager")

''' Nested Statement''' # statement inside the is called as nested statements

colour = input("enter color")
if colour == "yellow" :
    print("You're a CSK fan")
    num = int(input("enter number"))
    if num == 7 :
        print ("you're a MSD")
    else :
        print ("you're just a fan")
else :
    print ("You're a MI fan")
    num = int(input("enter number"))
    if num == 10 :
        print ("you're a RS fan")
    else:
        print("you're a bad MI fan")


""" HW : Create a conditonal statement for a for your +2 group selection """
   

"Loops"

#for loop - to repeat a fucntion over a values
range(10)

for i in range(5,10) : #iteration 
    print(i)

num = [80,90,40,50,60] #marks 100 - convert to 20

for i in num:
    print ((i/100)*20)


num = [80,90,40,50,60] #marks 100 - convert to 20

for i in num:
    print ((i*(.2)))

group_A = ["RCB","MI","CSK","DC"]

for team in group_A:
    print("group A :" ,team)
    
"""Nested for loop"""

group_B = ["SRH","KKR","RR","PKS"]

for teama in group_A:
    for teamb in group_B:
        print(teama, " vs " , teamb)

colum1=["orange","black","white"]
colum2=["chair","book","laptop"]
for i in colum1: 
    for j in colum2:
        print(i,j)

""" HW : Create a loops for set of mulitple students to +2 group selection """


group_A=["rcb","mi","csk","dc"]
for team in group_A:
    print("group A :" , team)
    
group_B=["srh","kkr","rr","pks"]
for teama in group_A:
    for teamb in group_B:
        print(teama, "vs" , teamb)


#break Statement - stop the loop with a condition 

group_A=["rcb","mi","csk","dc"]
for team in group_A:
    print("group A :" , team)
    if team == "csk":
        break


#continue statement - skip the loop with the condition 
#1
for team in group_A:
    if team == "csk":
        continue
    print("group A :" , team)

#2
for team in group_A:
    print("group A :" , team)
    if team == "csk":
        continue
    

#else in loops - used to complete the loop 
for team in group_A:
    print(team)
else:
    print("NO more Teams in A")

"""While Loop"""
i = 1
while i < 20 :
    print(i)
    
#break 
i=1
while i<10:
    print('number',i)
    i=i+1
    if (i==5):
        break

#continue 
i=0
while i < 10:
    i = i+1
    if (i==4):
        continue
    print(i)
else : 
    print("STOP!!")

stu = ["a","b","C"]
for i in stu :
    print ("name of student is :",i)
    Math = int(input("enter the marks of math : "))
    Science = int(input("enter the marks of science : "))
    English = int(input("enter the marks of english : "))
    if Math >= 75 and Science >= 75 and English >= 75:
        print("group_A")
    else:
        print("group_B")

''' HW: use while loop and for loop to loop the conditonal statements for choosing the group using the marks input'''

#pass statement - used to skip the unfinished loop or statement
for i in range(0,10) :
  pass


AJ = input()
AJ
