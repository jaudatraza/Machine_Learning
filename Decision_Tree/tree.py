#!/bin/env python
#-*- encoding: utf-8 -*-

from math import log, sqrt
import copy
import codecs

def count(dataset):	
	rst = dict()
	for data in dataset:
		label = data[-1]
		if label not in rst.keys(): 
			rst[label] = 0
		rst[label] += 1
	return rst

########################################################   
# Class for Decision Tree for Classifier
########################################################
class DecisionTree:

    def __init__(self, feature=-1, value=None, true_branch=None, 
                 false_branch=None, results={}, result=None, error=0):
        self.feature = feature
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        # Classification result at current node (majority class).
        # These three variables will change when building, evaluating or pruning a tree.
        self.result = result
        self.results = results
        self.error = error

    @classmethod
    ########################################################   
    # Split the True and False in the given feature
    ########################################################
    def _divide_set(cls, dataset, feature, value):
        if isinstance(value, int) or isinstance(value, float):
            func = lambda data: data[feature] >= value
        else:
            func = lambda data: data[feature] == value

        true_set = [data for data in dataset if func(data)]
        false_set = [data for data in dataset if not func(data)]
        return true_set, false_set

    @classmethod
    ########################################################   
    # Entropy measures impurity or disorder or uncertainty in the data.
    ########################################################
    
    def entropy(cls, dataset):
        log2 = lambda x: log(x) / log(2)
        rst = count(dataset)
        entropy = 0.0
	#Information = entropy(parent) - Weights Average * entropy(Children)
        for r in rst:
            p = float(rst[r]) / len(dataset)
            entropy -= p * log2(p)
        return entropy
    
    ########################################################   
    # Gini Index is calculated by subtracting the sum of the squared probabilities of each class 
    # from one
    ########################################################
    @classmethod
    def gini(cls, dataset):
        rst = count(dataset)
        gini = 1.0

        for r in rst:
            gini -= (rst[r] / len(dataset)) ** 2
        return gini

    ########################################################   
    # Using the function selected builds a trees, Iterative process for all the feautues. 
    ########################################################
    
    @classmethod
    def build_tree(cls, dataset, func):
        if len(dataset) == 0:
            return DecisionTree()

        best_gain = 0.0
        best_feature = None
        best_split = None
        cur_score = func(dataset)
        feature_cnt = len(dataset[0]) - 1

        results = count(dataset)
        result = sorted(results.items(), key=lambda x: x[1], reverse=True)[0][0]
        error = 0
        for k, v in results.items():
            if k != result:
                error += v

        # Choose the best feature
        for i in range(feature_cnt):

            unique_values = list(set([data[i] for data in dataset]))
            for v in unique_values:
                true_set, false_set = cls._divide_set(dataset, i, v)

                p_true = float(len(true_set)) / len(dataset)
                p_false = 1 - p_true
                gain = cur_score - p_true * \
                    func(true_set) - p_false * func(false_set)

                if gain > best_gain and len(true_set) and len(false_set):
                    best_gain = gain
                    best_feature = (i, v)
                    print("The best Feature till this run is:")
                    print(best_feature)
                    print("Best Gain Ratio")
                    print(best_gain)
                    best_split = (true_set, false_set)

        if not best_gain:
            return DecisionTree(result=result, results=results, error=error)

        true_branch = cls.build_tree(best_split[0], func)
        false_branch = cls.build_tree(best_split[1], func)
        print("************Best Feature to the tree************")
        print(best_feature[0])
        print(best_feature[1])
        #print("************Best Split to the tree************")
        #print(best_split[0])
        #print(best_split[1])
        return DecisionTree(feature=best_feature[0], value=best_feature[1], \
                    true_branch=true_branch, false_branch=false_branch, \
                    result=result, results=results, error=error)

    @classmethod
    ########################################################   
    # Prints out the tree in a seperate file with the heading, so could be read.
    ########################################################
    def plot_tree(cls, tree, headings, filepath=None):

        def _tree_to_str(tree, indent='\t\t'):
            # General output
            output = str(tree.result) + ' ' + str(tree.results) + \
                    ' err=' + str(tree.error)

            # Leaf node
            if not (tree.true_branch or tree.false_branch):
                return output

            if tree.feature in headings:
                col = headings[tree.feature]

            if isinstance(tree.value, int) or isinstance(tree.value, float):
                decision = ' %s >= %s ?' % (col, tree.value)
            else:
                decision = ' %s == %s ?' % (col, tree.value)

            true_branch = indent + 'yes -> ' + \
                _tree_to_str(tree.true_branch, indent + '\t\t')
            false_branch = indent + 'no  -> ' + \
                _tree_to_str(tree.false_branch, indent + '\t\t')
            return output + decision + '\n' + true_branch + '\n' + false_branch

        str_tree = _tree_to_str(tree)

        if filepath:
            with codecs.open(filepath, 'w', encoding='utf-8') as f:
                f.write(str_tree)
        else:
            print(str_tree)

    @classmethod
    ########################################################   
    # Evaluate how many of the of the rows in the given data predicted right
    ########################################################
    def evaluate(cls, tree, dataset):

        def _evaluate(eval_tree, dataset):
            eval_tree.results = count(dataset)
            eval_tree.error = 0
            for k, v in eval_tree.results.items():
                if k != eval_tree.result:
                    eval_tree.error += v

            # Leaf node
            if not (eval_tree.true_branch or eval_tree.false_branch):
                return eval_tree.error

            true_set = []
            false_set = []
            for data in dataset:
                v = data[eval_tree.feature]
                if isinstance(v, int) or isinstance(v, float):
                    if v >= eval_tree.value:
                        true_set.append(data)
                    else:
                        false_set.append(data)
                else:
                    if v == eval_tree.value:
                        true_set.append(data)
                    else:
                        false_set.append(data)
            return cls.evaluate(eval_tree.true_branch, true_set) + \
                    cls.evaluate(eval_tree.false_branch, false_set)

        # Deepcopy the tree to store test set info
        eval_tree = copy.deepcopy(tree)

        return _evaluate(eval_tree, dataset)

    @classmethod
    ########################################################   
    # Count the leaves in the Tree
    ########################################################
    def count_leaves(cls, tree):
        if not (tree.true_branch or tree.false_branch):
            return 1
        return cls.count_leaves(tree.true_branch) + cls.count_leaves(tree.false_branch)


    @classmethod
    
    ########################################################   
    # Bottom-up, left-to-right
    ########################################################
    def minimum_error_pruning(cls, tree):
        """
        Starting at the leaves, each node is replaced with its most popular class. If the prediction accuracy is not affected then the change is kept.
        """
        # Bottom-up, left-to-right
        sum_ = sum(tree.results.values())
        # (n(error) + k - 1) / n(all) + k
        error_leaf = (tree.error + 2) / (sum_ + 3)

        # Leaf node
        if not (tree.true_branch or tree.false_branch):
            return sum_, error_leaf

        sum_true, error_true = cls.minimum_error_pruning(tree.true_branch)
        sum_false, error_false = cls.minimum_error_pruning(tree.false_branch)
        error_subtree = sum_true / sum_ * error_true + sum_false / sum_ * error_false

        if error_leaf <= error_subtree:
            tree.true_branch = None
            tree.false_branch = None
            return sum_, error_leaf
        return sum_, error_subtree
