########################################################
# Author Jaudat Raza
# File Name = KNN_ALGO
# Description:
# This file has all the functions used to run the Project 2. 
# The implementation and code was ran on Jupyter Notebook, which is named
# Raza_Project2. It contains all the cell with their results. Also the 
# comments to explain the reason being some of the editing and functions. 
########################################################

########################################################
#	IMPORT LIBRARIES
########################################################
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import numpy as np
import scipy.stats as stats
import pandas as pd
import random
import math
from random import seed
from random import randrange
import time
import timeit

########################################################
#	LOAD DATA
########################################################
#Loading Abalone Data 
Abalone = pd.read_csv("abalone.data", sep = ',')
Abalone.columns = ["Sex", "Length","Diameter", "Height","WholeWeight","ShuckedWeight","VisceraWeight","ShellWeight", "Rings"]

#Loading Forest Fire Data
ForestFire = pd.read_csv("forestfires.data", sep = ',')
ForestFire.columns = ["X_Cord", "Y_Cord","Month", "Day","FFMC","DMC","DC","ISI", "Temp", "RH", "Wind", "Rain", "Area"]

#Loading Glass Data
Glass = pd.read_csv("glass.data", sep = ',')
Glass.columns = ["ID", "RefractiveIndex","Sodium", "Mag","Alum","Silicon","K","Calcium", "Barium", "Iron", "GlassType"]

#Loading House Vote
HouseVote = pd.read_csv("house-votes-84.data", sep = ',')
HouseVote.columns = ["Class", "Handicap","Water", "Adoption","Physician", "SalvadorAid", "ReligionInSchool", "SatelitteTestBan", "NicaraguanAid","Missile", "Immigration","CorpCutBack","Education","Sue","Crime","DutyFreeExport","ExportAdmin"]
HouseVote = HouseVote[["Handicap","Water", "Adoption","Physician", "SalvadorAid", "ReligionInSchool", "SatelitteTestBan", "NicaraguanAid","Missile", "Immigration","CorpCutBack","Education","Sue","Crime","DutyFreeExport","ExportAdmin", "Class"]]

#Loading Machine Data
Machine = pd.read_csv("machine.data", sep = ',')
Machine.columns = ["Vendor", "Model","MYCT", "MMIN","MMAX", "CACH", "CHMIN", "CHMAX", "PRP","ERP"]
Machine = Machine[["Model","MYCT", "MMIN","MMAX", "CACH", "CHMIN", "CHMAX", "PRP","ERP","Vendor"]]

#Load Image Segmentation Data and updating and sorting column
Segmentation = pd.read_csv("segmentation.data")
#Segmentation.columns = ["Region_COL", "Region_ROW","SLD5", "SLD2","VedgeMean", "VedgeSd", "HedgeMean", "HedgeSd", "IntensityMean","rawredM", "rawblueM", "rawgreenM","exredM", "exblueM", "exgreenM", "ValueM", "SatM", "HueM"]
Segmentation.to_csv('Segmentation.csv', header=False, index=False)
SegmentationC = pd.read_csv("Segmentation.csv")
OldColumn = list(Segmentation.columns)
Newcolumnw = ['REGION_CENTROID_COL','REGION_CENTROID_ROW','REGION_PIXEL_COUNT','SHORT_LINE_DENSITY_5','SHORT_LINE_DENSITY_2',
 'VEDGE_MEAN','VEDGE_SD', 'HEDGE_MEAN', 'HEDGE_SD', 'INTENSITY_MEAN', 'RAWRED_MEAN', 'RAWBLUE_MEAN',
 'RAWGREEN_MEAN', 'EXRED_MEAN', 'EXBLUE_MEAN', 'EXGREEN_MEAN', 'VALUE_MEAN', 'SATURATION_MEAN', 'HUE_MEAN']

SegmentationC.columns = Newcolumnw

########################################################
#	Data Cleanup
########################################################
########################################################
#	Function For Data Cleanup
########################################################
# Function goes through Data Frame and drop any row with ? mark in it
def DropDeadData(Data):
    NumberOfDataDropped = 0
    DataColums = list(Data.columns)
    for Attribute in DataColums:
        for i in Data.query(''+Attribute+' == "?"').index:
            Data=Data.drop(i)
            NumberOfDataDropped=NumberOfDataDropped + 1
    print(NumberOfDataDropped)

# Function replace text in Data Frame
def replace_Text(Data, Column):
    j = 1 
    for x in Column:
        Data = Data.replace(x, j)
        j = j+1 
    return Data
#Function split data 90 to 10%
def shuffle_split_data(X,val=10):
    arr_rand = np.random.rand(len(X))
    split = arr_rand < np.percentile(arr_rand, val)

    X_train = X[split]
    X_test =  X[~split]

    #print len(X_Train), len(X_Test)
    return X_test,X_train

########################################################   
#Abalone Data Clean up
########################################################
# create a list of our conditions
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
#House Vote Data Clean up
########################################################
# create a list of our conditions
conditions = [
    (HouseVote['Class'] == "democrat"),
    (HouseVote['Class'] == "republican")
    ]

# create a list of the values we want to assign for each condition
values = [1, 2]
# create a new column and use np.select to assign values to it using our lists as arguments
HouseVote['Class'] = np.select(conditions, values)
# Undecided Votes are marked as 2
HouseVote = HouseVote.replace('y',1)
HouseVote = HouseVote.replace('n',0)
HouseVote = HouseVote.replace('?',2)

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
#Forest Fire Data Clean up
########################################################
Days = ['mon', 'tue','wed', 'thu','fri', 'sat', 'sun']
Months = ['jan','feb','mar', 'apr', 'may','jun', 'jul','aug', 'sep', 'oct','nov','dec']

ForestFire = replace_Text(ForestFire, Days)
ForestFire = replace_Text(ForestFire, Months)

######################################################## 
########################################################   
# K Nearest Neighbor
########################################################
######################################################## 

# Transfer the dataframe to list
AbaloneData = Abalone.values.tolist()
ForestFireData = ForestFire.values.tolist()
GlassData = Glass.values.tolist()
HouseVoteData = HouseVote.values.tolist()
MachineData = Machine.values.tolist()
SegmentationCData = SegmentationC.values.tolist()

# Evaluate an algorithm using a cross validation split
#Parameter
#	dataset: list of data
#	algorithm: Which algorithm to use to make prediction
#	n_folds: Number of folds to split the data to
#	*args: Number of neighbors. 
#Output
#	Return score for all the prediction made and find the average accuracy 
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold) #
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores
    
# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
    
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split
   
# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)
# Main Function for implementing KNN   
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
 
# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction
 
# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return(predictions)
  
# Find the Best K by trying different K values till the "till" value  
def FindBestK(till, Data):
    Accuracy = 0
    BestK = 0
    for K in range(1,till):
        n_folds = 5
        scores = evaluate_algorithm(Data, k_nearest_neighbors, n_folds, K)
        j = (sum(scores)/float(len(scores)))
        if Accuracy <= j:
            BestK = K
            Accuracy = j
        scores = 0
    return BestK
 
# Find the Worst K by trying different K values till the "till" value  
def WorstK(till, Data):
    Accuracy = 100
    BestK = 0
    for K in range(1,till):
        n_folds = 5
        scores = evaluate_algorithm(Data, k_nearest_neighbors, n_folds, K)
        j = (sum(scores)/float(len(scores)))
        print(j)
        if Accuracy >= j:
            BestK = K
            Accuracy = j
        scores = 0
    return BestK
    
    
######################################################## 
########################################################   
# Edited K Nearest Neighbor
########################################################
######################################################## 

# Make a classification prediction with neighbors
def predict_classification_edited(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

def EditedKNN(PDDataset, K):
    J = 0
    K = 0
    DoneDropping = True
    while(DoneDropping):
        K = K+1
        J = 0
        DataList = PDDataset.values.tolist()
        PrviousJ = len(DataList)
#         print('Perevios J : '+ str(PrviousJ))
        for x in range(len(DataList)):
            prediction = predict_classification_edited(DataList, DataList[x], K)
            if DataList[x][-1] != prediction:
                PDDataset = PDDataset.drop(index = x)
                J = J + 1
#        print(J)
        if J < 1 or K > 10:
            DoneDropping = False    
        PDDataset = PDDataset.reset_index(drop=True)
                
    return PDDataset.values.tolist()


def ENNTOP(DataSet, Bestk, WorstK):
    EditedKNNData = EditedKNN(DataSet, WorstK)
    n_folds = 5
    num_neighbors = Bestk
    start = time.time()
    scores = evaluate_algorithm(EditedKNNData, k_nearest_neighbors, n_folds, num_neighbors)
    end = time.time()
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    Time = end - start
    print(Time)
    
######################################################## 
########################################################   
# Condensed K Nearest Neighbor
########################################################
######################################################## 

def CondensedKNN(PDDataset, K):
    CopyDataSet = []
    DataList = PDDataset.values.tolist()
    for x in range(len(DataList)-7):
        prediction = predict_classification_edited(DataList, DataList[x], K)
        if DataList[x][-1] == prediction:
            CopyDataSet.append(x)
            #print(CopyDataSet)
            #print(PDDataset.iloc[[x]])
            #time.sleep(0.1)
            #print(CopyDataSet)
            #PDDataset = PDDataset.drop(index = x)
    PDDataset = PDDataset.iloc[CopyDataSet]
    PDDataset = PDDataset.reset_index(drop=True)
    return PDDataset.values.tolist()

# def EditedCNN(PDDataset, K):
#     J = 0
#     K = 0
#     CopyDataSet = []
#     DoneDropping = True
#     while(DoneDropping):
#         K = K+1
#         J = 0
#         DataList = PDDataset.values.tolist()
#         PrviousJ = len(DataList)
# #         print('Perevios J : '+ str(PrviousJ))
#         for x in range(len(DataList)):
#             prediction = predict_classification_edited(DataList, DataList[x], K)
#             if DataList[x][-1] == prediction:
#                 CopyDataSet.append(x)
#                 #PDDataset = PDDataset.drop(index = x)
#                 J = J + 1
# #         print(J)
#         if J < 1 or K > 1:
#             DoneDropping = False    
#         PDDataset = PDDataset.reset_index(drop=True)
#     print(CopyDataSet)          
#     return PDDataset.values.tolist()

def CNN(DataSet,Bestk,WorsK):
    CondensedData = CondensedKNN(DataSet, 2)
    n_folds = 5
    num_neighbors = 5
    start = time.time()
    scores = evaluate_algorithm(CondensedData, k_nearest_neighbors, n_folds, num_neighbors)
    end = time.time()
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    Time = end - start
    print(Time)
    
