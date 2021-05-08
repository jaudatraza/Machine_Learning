import pandas as pd # Import Pandas library 
import numpy as np # Import Numpy library
 

 
def Winnow(Data, DataPath, TrainWeightFile, TrainOutFile, TestStatsFile, TestOutFile):
    # Required Data Set Format:
    # Columns (0 through N)
    # 0: Instance ID
    # 1: Attribute 1 (in binary)
    # 2: Attribute 2 (in binary)
    # 3: Attribute 3 (in binary)
    # ...
    # N: Actual Class (in binary)

    # This program then adds 8 additional columns:
    # N + 1: Weighted Sum (of the attributes)
    # N + 2: Predicted Class (in binary)...Weighted Sum > 0? (1 if yes; 0 if no)
    # N + 3: True Positive (1 if yes; O if no)
    # N + 4: False Positive (1 if yes; 0 if no)
    # N + 5: False Negative (1 if yes; 0 if no)
    # N + 6: True Negative (1 if yes; 0 if no)
    # N + 7: Promote (1 if yes; 0 if no) [for training set only]
    # N + 8: Demote (1 if yes; 0 if no)  [for training set only]

    ################ INPUT YOUR OWN VALUES IN THIS SECTION ######################
    ALGORITHM_NAME = "Winnow2"
    THETA = 0.5   # This is the threshold constant for the Winnow2 algorithm
    ALPHA = 2.0    # This is the adjustment constant for promotion & demotion
    DATA_PATH = DataPath  # Directory where data set is located
    TRAIN_WEIGHTS_FILE = TrainWeightFile # Weights of learned model
    TRAIN_OUT_FILE = TrainOutFile # Training phase of the model
    TEST_STATS_FILE = TestStatsFile # Testing statistics
    TEST_OUT_FILE = TestOutFile # Testing phase of the model
    TRAINING_DATA_PRCT = 0.67 # % of data set used for training
    testing_data_prct = 1 - TRAINING_DATA_PRCT # % of data set used for testing
    SEED = 99  # SEED for the random number generator. Default: 99
    #############################################################################


    

    # Create a training dataframe by sampling random instances from original data.
    # random_state guarantees that the pseudo-random number generator generates 
    # the same sequence of random numbers each time.
    pd_training_data = Data.sample(frac=TRAINING_DATA_PRCT, random_state=SEED)

    # Create a testing dataframe. Dropping the training data from the original
    # dataframe ensures training and testing dataframes have different instances
    pd_testing_data = Data.drop(pd_training_data.index)

    # Convert training dataframes to Numpy arrays
    np_training_data = pd_training_data.values
    np_testing_data = pd_testing_data.values


    ################ Begin Training Phase #####################################

    # Calculate the number of instances, columns, and attributes in the data set
    # Assumes 1 column for the instance ID and 1 column for the class
    # Record the index of the column that contains the actual class
    no_of_instances = np_training_data.shape[0]
    no_of_columns = np_training_data.shape[1]
    no_of_attributes = no_of_columns - 2
    actual_class_column = no_of_columns - 1

    # Initialize the weight vector. Initialize all weights to 1.
    # First column of weight vector is not used (i.e. Instance ID)
    weights = np.ones(no_of_attributes + 1)

    # Create a new array that has 8 columns, initialized to 99 for each value
    extra_columns_train = np.full((no_of_instances, 8),99)

    # Add extra columns to the training data set
    np_training_data = np.append(np_training_data, extra_columns_train, axis=1)

    # Make sure it is an array of floats
    np_training_data = np_training_data.astype(float)

    # Build the learning model one instance at a time
    for row in range(0, no_of_instances):

        # Set the weighted sum to 0
        weighted_sum = 0

        # Calculate the weighted sum of the attributes
        for col in range(1, no_of_attributes + 1):
            weighted_sum += (weights[col] * np_training_data[row,col])

        # Record the weighted sum into column N + 1, the column just to the right
        # of the actual class column
        np_training_data[row, actual_class_column + 1] = weighted_sum

        # Set the predicted class to 99
        predicted_class = 99

        # Learner's prediction: Is the weighted sum > THETA?
        if weighted_sum > THETA:
            predicted_class = 1
        else:
            predicted_class = 0

        # Record the predicted class into column N + 2
        np_training_data[row, actual_class_column + 2] = predicted_class

        # Record the actual class into a variable
        actual_class = np_training_data[row, actual_class_column]

        # Initialize the prediction outcomes
        # These variables are standard inputs into a "Confusion Matrix"
        true_positive = 0   # Predicted class = 1; Actual class = 1 (hit)
        false_positive = 0  # Predicted class = 1; Actual class = 0 (false alarm)
        false_negative = 0  # Predicted class = 0; Actual class = 1 (miss)
        true_negative = 0   # Predicted class = 0; Actual class = 0 

        # Determine the outcome of the Learner's prediction
        if predicted_class == 1 and actual_class == 1:
            true_positive = 1
        elif predicted_class == 1 and actual_class == 0:
            false_positive = 1
        elif predicted_class == 0 and actual_class == 1:
            false_negative = 1
        else:
            true_negative = 1

        # Record the outcome of the Learner's prediction
        np_training_data[row, actual_class_column + 3] = true_positive
        np_training_data[row, actual_class_column + 4] = false_positive
        np_training_data[row, actual_class_column + 5] = false_negative
        np_training_data[row, actual_class_column + 6] = true_negative

        # Set the promote and demote variables to 0
        promote = 0
        demote = 0

        # Promote if false negative
        if false_negative == 1:
            promote = 1

        # Demote if false positive
        if false_positive == 1:
            demote = 1

        # Record if either a promotion or demotion is needed
        np_training_data[row, actual_class_column + 7] = promote
        np_training_data[row, actual_class_column + 8] = demote

        # Run through each attribute and see if it is equal to 1
        # If attribute is 1, we need to either demote or promote (adjust the
        # corresponding weight by ALPHA).
        if demote == 1:
            for col in range(1, no_of_attributes + 1):
                if(np_training_data[row,col] == 1):
                    weights[col] /= ALPHA
        if promote == 1:
            for col in range(1, no_of_attributes + 1):
                if(np_training_data[row,col] == 1):
                    weights[col] *= ALPHA

    # Open a new file to save the weights
    outfile1 = open(TRAIN_WEIGHTS_FILE,"w") 

    # Write the weights of the Learned model to a file
    outfile1.write("----------------------------------------------------------\n")
    outfile1.write(" " + ALGORITHM_NAME + " Training Weights\n")
    outfile1.write("----------------------------------------------------------\n")
    outfile1.write("Data Set : " + DATA_PATH + "\n")
    outfile1.write("\n----------------------------\n")
    outfile1.write("Weights of the Learned Model\n")
    outfile1.write("----------------------------\n")
    for col in range(1, no_of_attributes + 1):
        colname = pd_training_data.columns[col]
        s = str(weights[col])
        outfile1.write(colname + " : " + s + "\n")

    # Write the relevant constants used in the model to a file
    outfile1.write("\n")
    outfile1.write("\n")
    outfile1.write("-----------\n")
    outfile1.write("Constants\n")
    outfile1.write("-----------\n")
    s = str(THETA)
    outfile1.write("THETA = " + s + "\n")
    s = str(ALPHA)
    outfile1.write("ALPHA = " + s + "\n")

    # Close the weights file
    outfile1.close()

    # Print the weights of the Learned model
    print("----------------------------------------------------------")
    print(" " + ALGORITHM_NAME + " Results")
    print("----------------------------------------------------------")
    print("Data Set : " + DATA_PATH)
    print()
    print()
    print("----------------------------")
    print("Weights of the Learned Model")
    print("----------------------------")
    for col in range(1, no_of_attributes + 1):
        colname = pd_training_data.columns[col]
        s = str(weights[col])
        print(colname + " : " + s)

    # Print the relevant constants used in the model
    print()
    print()
    print("-----------")
    print("Constants")
    print("-----------")
    s = str(THETA)
    print("THETA = " + s)
    s = str(ALPHA)
    print("ALPHA = " + s)
    print()

    # Print the learned model runs in binary form
    print("-------------------------------------------------------")
    print("Learned Model Runs of the Training Data Set (in binary)")
    print("-------------------------------------------------------")
    print(np_training_data)
    print()
    print()

    # Convert Numpy array to a dataframe
    df = pd.DataFrame(data=np_training_data)

    # Replace 0s and 1s in the attribute columns with False and True
    for col in range(1, no_of_attributes + 1):
        df[[col]] = df[[col]].replace([0,1],["False","True"])

    # Replace values in Actual Class column with more descriptive values
    #df[[actual_class_column]] = df[[actual_class_column]].replace([0,1],[CLASS_IF_ZERO,CLASS_IF_ONE])

    # Replace values in Predicted Class column with more descriptive values
    #df[[actual_class_column + 2]] = df[[actual_class_column + 2]].replace([0,1],[CLASS_IF_ZERO,CLASS_IF_ONE])

    # Change prediction outcomes to more descriptive values
    for col in range(actual_class_column + 3,actual_class_column + 9):
        df[[col]] = df[[col]].replace([0,1],["No","Yes"])

    # Rename the columns
    df.rename(columns={actual_class_column + 1 : "Weighted Sum" }, inplace = True)
    df.rename(columns={actual_class_column + 2 : "Predicted Class" }, inplace = True)
    df.rename(columns={actual_class_column + 3 : "True Positive" }, inplace = True)
    df.rename(columns={actual_class_column + 4 : "False Positive" }, inplace = True)
    df.rename(columns={actual_class_column + 5 : "False Negative" }, inplace = True)
    df.rename(columns={actual_class_column + 6 : "True Negative" }, inplace = True)
    df.rename(columns={actual_class_column + 7 : "Promote" }, inplace = True)
    df.rename(columns={actual_class_column + 8 : "Demote" }, inplace = True)

    # Change remaining columns names from position numbers to descriptive names
    for pos in range(0,actual_class_column + 1):
        df.rename(columns={pos : Data.columns[pos] }, inplace = True)

    print("-------------------------------------------------------")
    print("Learned Model Runs of the Training Data Set (readable) ")
    print("-------------------------------------------------------")
    # Print the revamped dataframe
    print(df)

    # Write revamped dataframe to a file
    df.to_csv(TRAIN_OUT_FILE, sep=",", header=True)
    ################ End Training Phase #####################################

    ################ Begin Testing Phase ######################################

    # Calculate the number of instances, columns, and attributes in the data set
    # Assumes 1 column for the instance ID and 1 column for the class
    # Record the index of the column that contains the actual class
    no_of_instances = np_testing_data.shape[0]
    no_of_columns = np_testing_data.shape[1]
    no_of_attributes = no_of_columns - 2
    actual_class_column = no_of_columns - 1

    # Create a new array that has 6 columns, initialized to 99 for each value
    extra_columns_test = np.full((no_of_instances, 6),99)

    # Add extra columns to the testing data set
    np_testing_data = np.append(np_testing_data, extra_columns_test, axis=1)

    # Make sure it is an array of floats
    np_testing_data = np_testing_data.astype(float)

    # Test the learning model one instance at a time
    for row in range(0, no_of_instances):

        # Set the weighted sum to 0
        weighted_sum = 0

        # Calculate the weighted sum of the attributes
        for col in range(1, no_of_attributes + 1):
            weighted_sum += (weights[col] * np_testing_data[row,col])

        # Record the weighted sum into column N + 1, the column just to the right
        # of the actual class column
        np_testing_data[row, actual_class_column + 1] = weighted_sum

        # Set the predicted class to 99
        predicted_class = 99

        # Learner's prediction: Is the weighted sum > THETA?
        if weighted_sum > THETA:
            predicted_class = 1
        else:
            predicted_class = 0

        # Record the predicted class into column N + 2
        np_testing_data[row, actual_class_column + 2] = predicted_class

        # Record the actual class into a variable
        actual_class = np_testing_data[row, actual_class_column]

        # Initialize the prediction outcomes
        # These variables are standard inputs into a "Confusion Matrix"
        true_positive = 0   # Predicted class = 1; Actual class = 1 (hit)
        false_positive = 0  # Predicted class = 1; Actual class = 0 (false alarm)
        false_negative = 0  # Predicted class = 0; Actual class = 1 (miss)
        true_negative = 0   # Predicted class = 0; Actual class = 0 

        # Determine the outcome of the Learner's prediction
        if predicted_class == 1 and actual_class == 1:
            true_positive = 1
        elif predicted_class == 1 and actual_class == 0:
            false_positive = 1
        elif predicted_class == 0 and actual_class == 1:
            false_negative = 1
        else:
            true_negative = 1

        # Record the outcome of the Learner's prediction
        np_testing_data[row, actual_class_column + 3] = true_positive
        np_testing_data[row, actual_class_column + 4] = false_positive
        np_testing_data[row, actual_class_column + 5] = false_negative
        np_testing_data[row, actual_class_column + 6] = true_negative

    # Convert Numpy array to a dataframe
    df = pd.DataFrame(data=np_testing_data)

    # Replace 0s and 1s in the attribute columns with False and True
    for col in range(1, no_of_attributes + 1):
        df[[col]] = df[[col]].replace([0,1],["False","True"])


    # Change prediction outcomes to more descriptive values
    for col in range(actual_class_column + 3,actual_class_column + 7):
        df[[col]] = df[[col]].replace([0,1],["No","Yes"])

    # Rename the columns
    df.rename(columns={actual_class_column + 1 : "Weighted Sum" }, inplace = True)
    df.rename(columns={actual_class_column + 2 : "Predicted Class" }, inplace = True)
    df.rename(columns={actual_class_column + 3 : "True Positive" }, inplace = True)
    df.rename(columns={actual_class_column + 4 : "False Positive" }, inplace = True)
    df.rename(columns={actual_class_column + 5 : "False Negative" }, inplace = True)
    df.rename(columns={actual_class_column + 6 : "True Negative" }, inplace = True)

    df_numerical = pd.DataFrame(data=np_testing_data) # Keep the values in this dataframe numerical
    df_numerical.rename(columns={actual_class_column + 3 : "True Positive" }, inplace = True)
    df_numerical.rename(columns={actual_class_column + 4 : "False Positive" }, inplace = True)
    df_numerical.rename(columns={actual_class_column + 5 : "False Negative" }, inplace = True)
    df_numerical.rename(columns={actual_class_column + 6 : "True Negative" }, inplace = True)

    # Change remaining columns names from position numbers to descriptive names
    for pos in range(0,actual_class_column + 1):
        df.rename(columns={pos : Data.columns[pos] }, inplace = True)

    print("-------------------------------------------------------")
    print("Learned Model Predictions on Testing Data Set")
    print("-------------------------------------------------------")
    # Print the revamped dataframe
    print(df)

    # Write revamped dataframe to a file
    df.to_csv(TEST_OUT_FILE, sep=",", header=True)

    # Open a new file to save the summary statistics
    outfile2 = open(TEST_STATS_FILE,"w") 

    # Write to a file
    outfile2.write("----------------------------------------------------------\n")
    outfile2.write(ALGORITHM_NAME + " Summary Statistics (Testing)\n")
    outfile2.write("----------------------------------------------------------\n")
    outfile2.write("Data Set : " + DATA_PATH + "\n")

    # Write the relevant stats to a file
    outfile2.write("\n")
    outfile2.write("Number of Test Instances : " +
        str(np_testing_data.shape[0])+ "\n")

    tp = df_numerical["True Positive"].sum()
    s = str(int(tp))
    outfile2.write("True Positives : " + s + "\n")

    fp = df_numerical["False Positive"].sum()
    s = str(int(fp))
    outfile2.write("False Positives : " + s + "\n")

    fn = df_numerical["False Negative"].sum()
    s = str(int(fn))
    outfile2.write("False Negatives : " + s + "\n")

    tn = df_numerical["True Negative"].sum()
    s = str(int(tn))
    outfile2.write("True Negatives : " + s + "\n")

    accuracy = (tp + tn)/(tp + tn + fp + fn)
    accuracy *= 100
    s = str(accuracy)
    outfile2.write("Accuracy : " + s + "%\n")


    # Close the weights file
    outfile2.close()

    # Print statistics to console
    print()
    print()
    print("-------------------------------------------------------")
    print(ALGORITHM_NAME + " Summary Statistics (Testing)")
    print("-------------------------------------------------------")
    print("Data Set : " + DATA_PATH)

    # Print the relevant stats to the console
    print()
    print("Number of Test Instances : " +
        str(np_testing_data.shape[0]))

    s = str(int(tp))
    print("True Positives : " + s)

    s = str(int(fp))
    print("False Positives : " + s)

    s = str(int(fn))
    print("False Negatives : " + s)

    s = str(int(tn))
    print("True Negatives : " + s)

    s = str(accuracy)
    print("Accuracy : " + s + "%")


    ###################### End Testing Phase ######################################
