#!/bin/env python
#-*- encoding: utf-8 -*-

import random
import conf
import csv

from tree import DecisionTree
########################################################   
# Function for Decision Tree for Classifier
########################################################
# This function. Reads the CSV file and return the header of the file and the Data
def load_dataset(DataPath):
    """
    Argument: DataPath: This is Data csv file location
    Return: Data and Header of the data
    """	
    reader = csv.reader(open(DataPath, 'rt'))

    headings = dict()
    for i, heading in enumerate(next(reader)):
        headings[i] = str(heading)

    dataset = [[convert_type(item) for item in row] for row in reader]
    return (headings, dataset)


def FiveFold(dataset):
	"""
	Argument: Data set 
	Procudure: Splits the Data into 5 different set. Each 20% of the recieved set
	"""

	Fold1 = dataset[:int(len(dataset)*0.20)]
	Fold2 = dataset[int(len(dataset)*0.20):int(len(dataset)*0.40)]
	Fold3 = dataset[int(len(dataset)*0.40):int(len(dataset)*0.60)]
	Fold4 = dataset[int(len(dataset)*0.60):int(len(dataset)*0.80)]
	Fold5 = dataset[int(len(dataset)*0.80):len(dataset)]
	return Fold1, Fold2, Fold3, Fold4, Fold5

def count(dataset):
    rst = dict()

    for data in dataset:
        label = data[-1]
        if label not in rst.keys(): 
            rst[label] = 0
        rst[label] += 1
    return rst


def convert_type(s):
    s = s.strip()

    try:
        return float(s) if '.' in s else int(s)
    except ValueError:
        return s
 
########################################################   
# Main Call function for Decision Tree for Classifier
########################################################            
def DTClass(CSVFile,UPT,PT):

	"""
	Argument
	CSVFile File Path to Data
	UPT is the file that will be created and contain Original Tree
	PT is the file that will be created and contain Prunned Tree
	Procudure: Prepare Data, Split, Build Original Tree for all five fold, Test the result, Prune the tree and then test it again. Prune using Minimum Error.  
	"""
	# Steps to build and prune a decision tree:

	# 1. Prepare dataset.
	headings, dataset = load_dataset(CSVFile)
	random.shuffle(dataset)
	# Split the dataset into training data, test data and pruning data if needed.
	train_data = dataset[:int(len(dataset)*0.90)]
	test_data = dataset[int(len(dataset)*0.90):len(dataset)]
	# prune_data = dataset[:]
	print(len(train_data))
	print(len(test_data))

	Fold1, Fold2, Fold3, Fold4, Fold5 = FiveFold(train_data)
	Folds = [Fold1, Fold2, Fold3, Fold4, Fold5]
	j = 1
	for Fold in Folds:
	    # 2. Grow a decision tree from training data based on Information Gain or gini.
	    dt = DecisionTree.build_tree(Fold, DecisionTree.gini)
	    # dt = DecisionTree.build_tree(train_data, DecisionTree.gini)
	    print('=======Fold Completed=======')		
	    TreeName = UPT+'_Fold'+str(j)
	    # 3. Visualize the tree.
	    DecisionTree.plot_tree(dt, headings, TreeName)
	    leaves = DecisionTree.count_leaves(dt)
	    print('Leaves count before pruning: %d' % leaves)


	    # 4. Run the test data through the tree.
	    err = DecisionTree.evaluate(dt, test_data)
	    print('Accuracy before pruning: %d/%d = %f' % \
		(len(test_data) - err, len(test_data), (len(test_data) - err) / len(test_data)))
	    j = j+1

	# 5. Prune the tree.
	DecisionTree.minimum_error_pruning(dt)


	# 6. Visualize the pruned tree.
	DecisionTree.plot_tree(dt, headings, PT)
	leaves = DecisionTree.count_leaves(dt)
	print('Leaves count after pruning: %d' % leaves)


	# 7. Check if the classification ability is improved after pruning.
	err = DecisionTree.evaluate(dt, test_data)
	print('Accuracy after pruning: %d/%d = %f' % \
	    (len(test_data) - err, len(test_data), (len(test_data) - err) / len(test_data)))


