#!/bin/env python
#-*- encoding: utf-8 -*-
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
import time
import timeit

import DTRegressor as DTRegressor
import DTClassifier as DTClassifier

########################################################
#	LOAD DATA
########################################################
#Abalone Data
Abalone = pd.read_csv("abalone.data", sep = ',')
Abalone.columns = ["Sex", "Length","Diameter", "Height","WholeWeight","ShuckedWeight","VisceraWeight","ShellWeight", "Rings"]

# ForestFire Data
ForestFire = pd.read_csv("forestfires.data", sep = ',')
ForestFire.columns = ["X_Cord", "Y_Cord","Month", "Day","FFMC","DMC","DC","ISI", "Temp", "RH", "Wind", "Rain", "Area"]

Segmentation = pd.read_csv("segmentation.data")
#Segmentation.columns = ["Region_COL", "Region_ROW","SLD5", "SLD2","VedgeMean", "VedgeSd", "HedgeMean", "HedgeSd", "IntensityMean","rawredM", "rawblueM", "rawgreenM","exredM", "exblueM", "exgreenM", "ValueM", "SatM", "HueM"]
Segmentation.to_csv('Segmentation.csv', header=False, index=False)
SegmentationC = pd.read_csv("Segmentation.csv")
OldColumn = list(Segmentation.columns)
Newcolumnw = ['REGION_CENTROID_COL','REGION_CENTROID_ROW','REGION_PIXEL_COUNT',
 'SHORT_LINE_DENSITY_5',
 'SHORT_LINE_DENSITY_2',
 'VEDGE_MEAN',
 'VEDGE_SD',
 'HEDGE_MEAN',
 'HEDGE_SD',
 'INTENSITY_MEAN',
 'RAWRED_MEAN',
 'RAWBLUE_MEAN',
 'RAWGREEN_MEAN',
 'EXRED_MEAN',
 'EXBLUE_MEAN',
 'EXGREEN_MEAN',
 'VALUE_MEAN',
 'SATURATION_MEAN',
 'HUE_MEAN']

SegmentationC.columns = Newcolumnw

Car = pd.read_csv("car.data")
Car.columns = ["Buying","Maint","Doors","Person","LugBoot","Safety", "Class"]


BreastCancer = pd.read_csv("breast-cancer-wisconsin.data", sep = ',')
BreastCancer.columns = ["Sample_code_number", "Clump_Thickness","Uniformity_CellSize", "Uniformity_CellShape","MarginalAdhesion","SE_CellSize","BareNuclei","BlandChromatin", "NormalNucleoli", "Mitoses", "Class"]

Machine = pd.read_csv("machine.data", sep = ',')
Machine.columns = ["Vendor", "Model","MYCT", "MMIN","MMAX", "CACH", "CHMIN", "CHMAX", "PRP","ERP"]
Machine = Machine[["Model","MYCT", "MMIN","MMAX", "CACH", "CHMIN", "CHMAX", "PRP","ERP","Vendor"]]

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
#Abalone Data Clean up
########################################################
conditions = [
    (Abalone['Sex'] == "I"),
    (Abalone['Sex'] == "F"),
    (Abalone['Sex'] == "M")
    ]

# create a list of the values we want to assign for each condition
values = [1, 2, 3]

# create a new column and use np.select to assign values to it using our lists as arguments
Abalone['Sex'] = np.select(conditions, values)
########################################################   
#Image Segmentation Data Clean up and CSV File
########################################################

SegmentationC.to_csv('ImageSegmentation.csv', index = False) 

########################################################   
#Forest Fire Data Clean up
########################################################
Days = ['mon', 'tue','wed', 'thu','fri', 'sat', 'sun']
Months = ['jan','feb','mar', 'apr', 'may','jun', 'jul','aug', 'sep', 'oct','nov','dec']

ForestFire = replace_Text(ForestFire, Days)
ForestFire = replace_Text(ForestFire, Months)

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

BreastCancer.to_csv('BreastCancer.csv', index = False) 

########################################################   
#Machine Data Clean up
########################################################
Vendors = ['amdahl', 'apollo', 'basf', 'bti', 'burroughs', 'c.r.d', 'cdc',
       'cambex', 'dec', 'dg', 'formation', 'four-phase', 'gould', 'hp',
       'harris', 'honeywell', 'ibm', 'ipl', 'magnuson', 'microdata',
       'nas', 'ncr', 'nixdorf', 'perkin-elmer', 'prime', 'siemens',
       'sperry', 'sratus', 'wang']

Models = ['470v/7', '470v/7a', '470v/7b', '470v/7c', '470v/b', '580-5840',
       '580-5850', '580-5860', '580-5880', 'dn320', 'dn420', '7/65',
       '7/68', '5000', '8000', 'b1955', 'b2900', 'b2925', 'b4955',
       'b5900', 'b5920', 'b6900', 'b6925', '68/10-80', 'universe:2203t',
       'universe:68', 'universe:68/05', 'universe:68/137',
       'universe:68/37', 'cyber:170/750', 'cyber:170/760',
       'cyber:170/815', 'cyber:170/825', 'cyber:170/835', 'cyber:170/845',
       'omega:480-i', 'omega:480-ii', 'omega:480-iii', '1636-1',
       '1636-10', '1641-1', '1641-11', '1651-1', 'decsys:10:1091',
       'decsys:20:2060', 'microvax-1', 'vax:11/730', 'vax:11/750',
       'vax:11/780', 'eclipse:c/350', 'eclipse:m/600', 'eclipse:mv/10000',
       'eclipse:mv/4000', 'eclipse:mv/6000', 'eclipse:mv/8000',
       'eclipse:mv/8000-ii', 'f4000/100', 'f4000/200', 'f4000/200ap',
       'f4000/300', 'f4000/300ap', '2000/260', 'concept:32/8705',
       'concept:32/8750', 'concept:32/8780', '3000/30', '3000/40',
       '3000/44', '3000/48', '3000/64', '3000/88', '3000/iii', '100',
       '300', '500', '600', '700', '80', '800', 'dps:6/35', 'dps:6/92',
       'dps:6/96', 'dps:7/35', 'dps:7/45', 'dps:7/55', 'dps:7/65',
       'dps:8/44', 'dps:8/49', 'dps:8/50', 'dps:8/52', 'dps:8/62',
       'dps:8/20', '3033:s', '3033:u', '3081', '3081:d', '3083:b',
       '3083:e', '370/125-2', '370/148', '370/158-3', '38/3', '38/4',
       '38/5', '38/7', '38/8', '4321', '4331-1', '4331-11', '4331-2',
       '4341', '4341-1', '4341-10', '4341-11', '4341-12', '4341-2',
       '4341-9', '4361-4', '4361-5', '4381-1', '4381-2', '8130-a',
       '8130-b', '8140', '4436', '4443', '4445', '4446', '4460', '4480',
       'm80/30', 'm80/31', 'm80/32', 'm80/42', 'm80/43', 'm80/44',
       'seq.ms/3200', 'as/3000', 'as/3000-n', 'as/5000', 'as/5000-e',
       'as/5000-n', 'as/6130', 'as/6150', 'as/6620', 'as/6630', 'as/6650',
       'as/7000', 'as/7000-n', 'as/8040', 'as/8050', 'as/8060',
       'as/9000-dpc', 'as/9000-n', 'as/9040', 'as/9060', 'v8535:ii',
       'v8545:ii', 'v8555:ii', 'v8565:ii', 'v8565:ii-e', 'v8575:ii',
       'v8585:ii', 'v8595:ii', 'v8635', 'v8650', 'v8655', 'v8665',
       'v8670', '8890/30', '8890/50', '8890/70', '3205', '3210', '3230',
       '50-2250', '50-250-ii', '50-550-ii', '50-750-ii', '50-850-ii',
       '7.521', '7.531', '7.536', '7.541', '7.551', '7.561', '7.865-2',
       '7.870-2', '7.872-2', '7.875-2', '7.880-2', '7.881-2',
       '1100/61-h1', '1100/81', '1100/82', '1100/83', '1100/84',
       '1100/93', '1100/94', '80/3', '80/4', '80/5', '80/6', '80/8',
       '90/80-model-3', '32', 'vs-100', 'vs-90']


Machine = replace_Text(Machine, Vendors)   
Machine = replace_Text(Machine, Models) 

########################################################   
#Car Data Clean up
########################################################
#Car["Class"].unique()
Buying = ['vhigh', 'high', 'med', 'low']
Maint = ['vhigh', 'high', 'med', 'low']
Doors = ['2', '3', '4', '5more']
Person = ['2', '4', 'more']
LugBoot = ['small', 'med', 'big']
Safety = ['med', 'high', 'low']
Class = ['unacc', 'acc', 'vgood', 'good']

Car = replace_Text(Car, Buying)   
Car = replace_Text(Car, Maint) 
Car = replace_Text(Car, Doors, starting = 2) 
Car = replace_Text(Car, Person) 
Car = replace_Text(Car, LugBoot) 
Car = replace_Text(Car, Safety)
Car = replace_Text(Car, Class)

Car.to_csv('CarEvaluation.csv', index = False) 

########################################################   
#Running Regressor Tree Forest Fire
########################################################
# Creates Original and Cut off Threshold of 100  for early stopping
# Default to 0, which means no early stopping 237 Original Depth
DTRegressor.DTReg(ForestFire, 100)

########################################################   
#Running Regressor Tree Abalone
########################################################
# Creates Original and Cut off Threshold of 450  for early stopping
# Default to 0, which means no early stopping 2251 Original Depth

DTRegressor.DTReg(Abalone, 450)

########################################################   
#Running Regressor Tree Computer Hardware
########################################################
# Creates Original and Cut off Threshold of 70  for early stopping
# Default to 0, which means no early stopping

DTRegressor.DTReg(Machine, 70)

########################################################   
#Running Classification Tree Breast Cancer
########################################################
# Creates Original and Pruned tree which are printed in the file given as parameter
# Uses information Gain to for splitting crierion
DTClassifier.DTClass('BreastCancer.csv', 'BC_OriginalT', 'BC_PrunnedT')

########################################################   
#Running Classification Tree Image Segmentation
########################################################
# Creates Original and Pruned tree which are printed in the file given as parameter
# Uses information Gain to for splitting crierion
DTClassifier.DTClass('ImageSegmentation.csv', 'IS_OriginalT', 'IS_PrunnedT')

########################################################   
#Running Classification Tree Car Evaluation
########################################################
# Creates Original and Pruned tree which are printed in the file given as parameter
# Uses information Gain to for splitting crierion

DTClassifier.DTClass('CarEvaluation.csv', 'CE_OriginalT', 'CE_PrunnedT')
