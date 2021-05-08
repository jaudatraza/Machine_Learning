import pandas as pd # Import Pandas library 
import numpy as np # Import Numpy library
import five_fold_stratified_cv
import naive_bayes
 
# Description: Main File to Run Bayes Algorithm 

 
# Can only take dsicrete data set. Data preprocessed for that purpose. 
 
# The naive_bayes.py program then adds 2 additional columns for the test set.
# N + 1: Predicted Class
# N + 2: Prediction Correct? (1 if yes, 0 if no)
 
ALGORITHM_NAME = "Naive Bayes"
 
def BayesMain(Data, DataPath, TraceRunFile, StatsFile):
 
    print("" +  ALGORITHM_NAME + " Program!")
    print()
 
    # Directory where data set is located
    data_path = DataPath
 
    # Read the full text file and store records in a Pandas dataframe
    pd_data_set = Data
 
    # Show functioning of the program
    trace_runs_file = TraceRunFile
 
    # Open a new file to save trace runs
    outfile_tr = open(trace_runs_file,"w") 
 
    # Testing statistics
    test_stats_file = StatsFile
 
    # Open a test_stats_file 
    outfile_ts = open(test_stats_file,"w")
 
    # 5 folds in the cross-validation
    NO_OF_FOLDS = 5
 
    # Generate the five stratified folds using the function in Five fold stratified py file
    fold0, fold1, fold2, fold3, fold4 = five_fold_stratified_cv.get_five_folds(
        pd_data_set)
 
    training_dataset = None
    test_dataset = None
 
    # Create an empty array of length 5 to store the Stats_Accuracy 
    # (classification accuracy)
    Stats_Accuracy = np.zeros(NO_OF_FOLDS)
 
    # Run Naive Bayes the designated number of times as indicated by the 
    # number of folds
    for experiment in range(0, NO_OF_FOLDS):
 
        print()
        print("Running Experiment " + str(experiment + 1) + " ...")
        print()
        outfile_tr.write("Running Experiment " + str(experiment + 1) + " ...\n")
        outfile_tr.write("\n")
 
        # Each fold will have a chance to be the test data set
        if experiment == 0:
            test_dataset = fold0
            training_dataset = pd.concat([
               fold1, fold2, fold3, fold4], ignore_index=True, sort=False)                
        elif experiment == 1:
            test_dataset = fold1
            training_dataset = pd.concat([
               fold0, fold2, fold3, fold4], ignore_index=True, sort=False) 
        elif experiment == 2:
            test_dataset = fold2
            training_dataset = pd.concat([
               fold0, fold1, fold3, fold4], ignore_index=True, sort=False) 
        elif experiment == 3:
            test_dataset = fold3
            training_dataset = pd.concat([
               fold0, fold1, fold2, fold4], ignore_index=True, sort=False) 
        else:
            test_dataset = fold4
            training_dataset = pd.concat([
               fold0, fold1, fold2, fold3], ignore_index=True, sort=False) 
         
        # Run Naive Bayes
        accuracy, predictions, learned_model, no_of_instances_test = (
            naive_bayes.naive_bayes(training_dataset,test_dataset))
 
        # Replace 1 with Yes and 0 with No in the 'Prediction 
        # Correct?' column
        predictions['Prediction Correct?'] = predictions[
            'Prediction Correct?'].map({1: "Yes", 0: "No"})
 
        # Print the trace runs of each experiment
        print("Accuracy:")
        print(str(accuracy * 100) + "%")
        print()
        print("Classifications:")
        print(predictions)
        print()
        print("Learned Model (Likelihood Table):")
        print(learned_model)
        print()
        print("Number of Test Instances:")
        print(str(no_of_instances_test))
        print() 
 
        outfile_tr.write("Accuracy:")
        outfile_tr.write(str(accuracy * 100) + "%\n\n")
        outfile_tr.write("Classifications:\n")
        outfile_tr.write(str(predictions) + "\n\n")
        outfile_tr.write("Learned Model (Likelihood Table):\n")
        outfile_tr.write(str(learned_model) + "\n\n")
        outfile_tr.write("Number of Test Instances:")
        outfile_tr.write(str(no_of_instances_test) + "\n\n")
 
        # Store the accuracy in the Stats_Accuracy array
        Stats_Accuracy[experiment] = accuracy
 
    outfile_tr.write("Experiments Completed.\n")
    print("Experiments Completed.\n")
 
    # Write to a file
    outfile_ts.write("----------------------------------------------------------\n")
    outfile_ts.write(ALGORITHM_NAME + " Summary Statistics\n")
    outfile_ts.write("----------------------------------------------------------\n")
    outfile_ts.write("Data Set : " + data_path + "\n")
    outfile_ts.write("\n")
    outfile_ts.write("Accuracy Statistics for All 5 Experiments:")
    outfile_ts.write(np.array2string(
        Stats_Accuracy, precision=2, separator=',',
        suppress_small=True))
    outfile_ts.write("\n")
    outfile_ts.write("\n")
    accuracy = np.mean(Stats_Accuracy)
    accuracy *= 100
    outfile_ts.write("Classification Accuracy : " + str(accuracy) + "%\n")
    
    # Print to the console
    print()
    print("----------------------------------------------------------")
    print(ALGORITHM_NAME + " Summary Statistics")
    print("----------------------------------------------------------")
    print("Data Set : " + data_path)
    print()
    print()
    print("Accuracy Statistics for All 5 Experiments:")
    print(Stats_Accuracy)
    print()
    print()
    print("Classification Accuracy : " + str(accuracy) + "%")
    print()
 
    # Close the files
    outfile_tr.close()
    outfile_ts.close()
 
