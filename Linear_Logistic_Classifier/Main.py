#!/bin/env python
#-*- encoding: utf-8 -*-

#Jaudat Raza

# Main show the result of running on Jypyter Notebook. 
# Then the code is moved into the py files. 

########################################################
#	IMPORT LIBRARIES
########################################################
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Imports 
import numpy as np
import scipy.stats as stats
# import seaborn as sns
# import matplotlib.pyplot as plt
import pandas as pd
import random
import math
from random import seed
from random import randrange
import scipy.optimize as opt
from tabulate import tabulate
import time
import timeit

import MultiClassLogisticRegression 
import LogisticRegression
import Adaline

########################################################
#	LOAD DATA
########################################################
BreastCancer = pd.read_csv("breast-cancer-wisconsin.data", sep = ',')
BreastCancer.columns = ["Sample_code_number", "Clump_Thickness","Uniformity_CellSize", "Uniformity_CellShape","MarginalAdhesion","SE_CellSize","BareNuclei","BlandChromatin", "NormalNucleoli", "Mitoses", "Class"]

Glass = pd.read_csv("glass.data", sep = ',')
Glass.columns = ["ID", "RefractiveIndex","Sodium", "Mag","Alum","Silicon","K","Calcium", "Barium", "Iron", "GlassType"]
                 
Iris = pd.read_csv("iris.data", sep = ',')
Iris.columns = ["Sepal_L", "Sepal_W","Petal_L", "Petal_W","Class"]  
                 
SoyBean = pd.read_csv("soybean-small.data", sep = ',')   
Soycolumn = ['date',
'plantstand',
'precip',
'temp',
'hail',
'crophist',
'areadamaged',
'severity',
'seedtmt',
'germination',
'plantgrowth',
'leaves',
'leafspotshalo',
'leafspotsmarg',
'leafspotsize',
'leafshread',
'leafmalf',
'leafmild',
'stem',
'lodging',
'stemcankers',
'cankerlesion',
'fruitingbodies',
'externaldecay',
'mycelium',
'intdiscolor',
'sclerotia',
'fruitpods',
'fruitspots',
'seed',
'moldgrowth',
'seeddiscolor',
'seedsize',
'shriveling',
'roots','Distrubution']
SoyBean = pd.read_csv("soybean-small.data", sep = ',')
SoyBean.columns = Soycolumn

HouseVote = pd.read_csv("house-votes-84.data", sep = ',')
HouseVote.columns = ["Class", "Handicap","Water", "Adoption","Physician", "SalvadorAid", "ReligionInSchool", "SatelitteTestBan", "NicaraguanAid","Missile", "Immigration","CorpCutBack","Education","Sue","Crime","DutyFreeExport","ExportAdmin"]
HouseVote = HouseVote[["Handicap","Water", "Adoption","Physician", "SalvadorAid", "ReligionInSchool", "SatelitteTestBan", "NicaraguanAid","Missile", "Immigration","CorpCutBack","Education","Sue","Crime","DutyFreeExport","ExportAdmin", "Class"]]

########################################################
#	Data Cleanup
########################################################
########################################################
#	Function For Data Cleanup
########################################################

def replace_Text(Data, Column, starting = 1):
    """Replace unique value in order starting with Starting arugment"""
    j = starting 
    for x in Column:
        Data = Data.replace(x, j)
        j = j+1 
    return Data
    
########################################################   
#Breast Cancer Data Clean up
########################################################
NumberOfDataDropped = 0
for Attribute in ["Sample_code_number", "Clump_Thickness","Uniformity_CellSize", "Uniformity_CellShape","MarginalAdhesion","SE_CellSize","BareNuclei","BlandChromatin", "NormalNucleoli", "Mitoses", "Class"]:
    for i in BreastCancer.query(''+Attribute+' == "?"').index:
        BreastCancer=BreastCancer.drop(i)
        NumberOfDataDropped=NumberOfDataDropped + 1
        
BreastCancer['BareNuclei']=pd.to_numeric(BreastCancer['BareNuclei'])

# create a list of our conditions
conditions = [
    (BreastCancer['Class'] == 2),
    (BreastCancer['Class'] == 4)
    ]


# create a list of the values we want to assign for each condition
values = [1, 0]

# create a new column and use np.select to assign values to it using our lists as arguments
BreastCancer['Class'] = np.select(conditions, values)

BreastCancer=BreastCancer.drop("Sample_code_number",1)

BreastCancer.to_csv('BreastCancer.csv', index = False) 

########################################################   
#Glass Data Clean up
########################################################
GlassType = [1, 2, 3, 5, 6, 7]
Glass = replace_Text(Glass, GlassType)   

########################################################   
#Iris Data Clean up
########################################################
# create a list of our conditions
conditions = [
    (Iris['Class'] == "Iris-setosa"),
    (Iris['Class'] == "Iris-virginica"),
    (Iris['Class'] == "Iris-versicolor")
    ]

# create a list of the values we want to assign for each condition
values = [0, 1, 2]

# create a new column and use np.select to assign values to it using our lists as arguments
Iris['Class'] = np.select(conditions, values)

########################################################   
#Soybean Data Clean up
########################################################

SoycolumnCheck = ['date',
'plantstand',
'precip',
'temp',
'hail',
'crophist',
'areadamaged',
'severity',
'seedtmt',
'germination',
'plantgrowth',
'leaves',
'leafspotshalo',
'leafspotsmarg',
'leafspotsize',
'leafshread',
'leafmalf',
'leafmild',
'stem',
'lodging',
'stemcankers',
'cankerlesion',
'fruitingbodies',
'externaldecay',
'mycelium',
'intdiscolor',
'sclerotia',
'fruitpods',
'fruitspots',
'seed',
'moldgrowth',
'seeddiscolor',
'seedsize',
'shriveling',
'roots']
for Attribute in SoycolumnCheck:
    if SoyBean[Attribute].mean() == 0.0:
        #print(Attribute)
        SoyBean = SoyBean.drop(columns = Attribute)
        
# create a list of our conditions
conditions = [
    (SoyBean['Distrubution'] == "D1"),
    (SoyBean['Distrubution'] == "D2"),
    (SoyBean['Distrubution'] == "D3"),
    (SoyBean['Distrubution'] == "D4")
    ]

# create a list of the values we want to assign for each condition
values = [0, 1, 2, 3]

# create a new column and use np.select to assign values to it using our lists as arguments
SoyBean['Distrubution'] = np.select(conditions, values)

########################################################   
#House Vote Data Clean up
########################################################
########################################################   
#House Vote Data Clean up
########################################################
conditions = [
    (HouseVote['Class'] == "democrat"),
    (HouseVote['Class'] == "republican")
    ]

# create a list of the values we want to assign for each condition
values = [0, 1]

# create a new column and use np.select to assign values to it using our lists as arguments
HouseVote['Class'] = np.select(conditions, values)

HouseVote = HouseVote.replace('y',1)
HouseVote = HouseVote.replace('n',0)
HouseVote = HouseVote.replace('?',2)


######################################################## 
######################################################## 
######################################################## 
######################################################## 
# Call this function after calling the parent file

########################################################   
#Running Log Classification on Iris Data
########################################################
# Log Class main call
MulticlassLogReg(Iris,'Sepal_L', 'Sepal_W')

########################################################   
#Running Log Classification on HouseVote Data
########################################################
# Log Class main call
MulticlassLogReg(HouseVote, "Education", "ReligionInSchool")

########################################################   
#Running Log Classification on BreastCancer Data
########################################################
# Log Class main call

MulticlassLogReg(BreastCancer, "Clump_Thickness", "Uniformity_CellSize")

########################################################   
#Running Log Classification on Glass Data
########################################################
# Log Class main call

MulticlassLogReg(Glass, "Calcium", "Mag")     

########################################################   
#Running Log Classification on SoyBean Data
########################################################
# Log Class main call

MulticlassLogReg(SoyBean, "precip", "temp")    

######################################################## 
######################################################## 
######################################################## 
# Call this function after calling the Adaline File function
Adaline(BreastCancer)
Adaline(Iris)
Adaline(HouseVote)
Adaline(SoyBean)
Adaline(Glass)

######################################################## 
######################################################## 
######################################################## 
# Logestic Classification one by one feature
LogisticRegressionRegular(Iris)
LogisticRegressionRegular(BreastCancer)
LogisticRegressionRegular(SoyBean)
LogisticRegressionRegular(HouseVote)
LogisticRegressionRegular(Glass)
