# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 18:26:32 2021

@author: AJ
"""

​
import numpy as np
​
# HW : Create a conditonal statement for a for your +2 group selection
​
Maths = int(input("\n Enter your marks in Mathematics "))
if Maths >= 70:
    sci = int(input("\n Enter your marks in Science "))
    if sci >=70:
        print("\n You can preferablly choose Science Stream")
    else:
        print("\n You can preferablly choose Commerse Stream")
else:
    lang_list = []
    for i in range(1,4):
        try:
            sub = int(input(f"\n Enter your marks in Language {i} "))
        except:
            continue
        lang_list.append(sub)
    avg_lang = np.mean(lang_list)
    if avg_lang >= 60:
        print("\n You can preferablly choose Arts Stream")
    else:
        print("\n You should choose some extra curicular activity")
​
# HW : Create a loops for set of mulitple students to +2 group selection
Students = {}
x = "Y"
while x == "Y":
    Name = input("\n Enter name of student: ")
    Maths = int(input("\n Enter your marks in Mathematics "))
    if Maths >= 70:
        sci = int(input("\n Enter your marks in Science "))
        if sci >=70:
            Students[Name] = "Science"
        else:
            Students[Name] = "Commerce"
    else:
        lang_list = []
        for i in range(1,4):
            try:
                sub = int(input(f"\n Enter your marks in Language {i} "))
            except:
                continue
            lang_list.append(sub)
        avg_lang = np.mean(lang_list)
        if avg_lang >= 60:
            Students[Name] = "Arts"
        else:
            Students[Name] = "Extra Activites"
    x = input("\n to continue enter Y ").upper()
print("\n List of Students with choices",Students)