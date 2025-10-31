# Peter Raatikainen 2018
# This script gives a result of different hyperparameter combinations and their scores
# by using a own gridsearch function with cross-validation

from time import time
import argparse
import itertools
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn import svm

import pandas as pd
import matplotlib.pyplot as plt
import collections # This is for Counter
from operator import itemgetter # This is for sorting the hyperparameter results
from itertools import chain
from matplotlib.ticker import NullFormatter

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix


# Allocate shorter names for the columns in the data file
#names = ['id', 'trialId', 'trialIndex', 'fixDuration', 'fixIndex', 'fixX', 'fixY', 'sacDuration', 'sacAmplitude', 'fixAreaIndex', 'interestAreaGroup', 'zPaf10']

# The Argument parser and arguments -----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cycles", help="The number of cycles to do", type=int, choices=range(1,100000))
parser.add_argument("-f", "--folds", help="The number of folds", type=int, choices=range(2,162))
parser.add_argument("-o", "--open", help="The file to open")
args = parser.parse_args()
# ---------------------------------------------------------------------------
trialRows = False


# Load the data from the file
df = pd.read_csv(args.open, sep=';', header=0, index_col="id", decimal=',')

#############################################################################################
#chosen = df[['dyslexia', 'T2FSaccDurBin1', 'T2FSaccAmpBin1', 'T2FFixDurBin5', 'T11FSaccAmpBin1',
 #            'T7DFixDurBin2', 'T8TSaccAmpBin1', 'T10FSaccAmpBin1', 'T2F-F', 'T3FSaccAmpBin1',
  #           'T5FSaccAmpBin1', 'T8F-L', 'T3FFixDurBin5', 'T10T-T', 'T9FSaccAmpBin2', 'T3LFixDurBin5',
  #           'T7DSaccAmpBin1', 'T3DSaccAmpBin1', 'T2TSaccAmpBin1', 'T4FSaccAmpBin1', 'T5FFixDurBin5']]

#chosen.to_csv("featuresOwn.csv", sep=";", index=True, index_label="id", decimal=',')
#print("...Written to file.")
#############################################################################################

#["meanFixDuration", "fixCount", "meanSacDuration", "dyslexia"]
# Input variables
X = df.drop("dyslexia", axis=1)

# Create empty dataframe just with indexes for tracking incorrectly predicted classes
indexX = pd.DataFrame(index=X.index)

# Print details of input variables
print(X.head())
print("Shape of X: ", X.shape)

# Output variables
y = np.array(df['dyslexia'])

# Print details of output variables
#print(y)
print("Shape of y: ", y.shape)

# This function prints the confusion matrix
def plot_confusion_matrix(cm, normalize=False):

    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix\n")
    else:
        print('Confusion matrix, without normalization\n')

    print(cm)

# The function for doing my own grid search
def gridSearch(clf, parameters, cv, scoring):

    #y_pred = model_selection.cross_val_predict(clf, X, y, cv=outerCV)

    cv = KFold(len(labels), n_folds=20)
    clf = SVC()

    # Go through all the splits
    for train_index, test_index in cv.split(X):
        clf.fit(X[train_index], labels[train_index])
        y_pred = clf.predict(X[test_index])

        # Calculate the confusion matrix
        confusion_matrix = confusion_matrix(labels[test_index], y_pred)

# Set the time
t0 = time()




# Find out the best parameters for training the SVM
print("\nPerforming parameter tuning")

folds = args.folds
print("Folds: ", folds)

scoring = "f1"
#{"F1": "f1",
#          "Precision": "precision",
#          "Recall": "recall"}
print("\nThe scoring used: ", scoring)

# NUMBER OF CYCLES
cycles = args.cycles

# List for storing results from the cycles
cycleScores = list()

# List for storing recall values for dyslexia from the cycles
foldRecalls = np.empty(folds)

# List for storing precision values for dyslexia from the cycles
foldPrecisions = np.empty(folds)

# List for storing recall values from the cycles
foldRecallsND = np.empty(folds)

# List for storing precision values from the cycles
foldPrecisionsND = np.empty(folds)

# The scores of the different hyperparameters
hyperparameterResults = list()

hyperparameterResultsTRows = list()

# This list contains all of the incorrectly predicted dyslectics
allIncorrect = list()

# List for storing all of the confusion matrices
confMatricesFolds = list()







# List for storing the best parameters from each cycle'
bestParams = list()
# Original grid stored here for historical detail (used while developing method):


# hyperParameters = {"C": [1000, 2000, 3000, 5000, 7000, 10000, 20000, 50000, 100000, 500000, 1000000],
#                    "gamma": [0.00004, 0.00006, 0.00008, 0.0001, 0.0005, 0.001, 0.005, 0.01]}

# Final documented results were computed by fine-tuning parameters here,
# after optimal range had been narrowed down using the grid search:
hyperParameters = {"C": [1],
                  "gamma": [0.1, 1]}
# (As an early preliminary step, we compared different kernels; the default RBF was found suitable)
hyperparameterComb = len(hyperParameters["C"]) * len(hyperParameters["gamma"])

# Create the arrays for all the hyperparameter results for both dyslectics and nondyslectics (=ND)
recallResults = np.empty((hyperparameterComb, cycles*folds))
precisionResults = np.empty((hyperparameterComb, cycles*folds))
recallResultsND = np.empty((hyperparameterComb, cycles*folds))
precisionResultsND = np.empty((hyperparameterComb, cycles*folds))
accuracyResults = np.empty((hyperparameterComb, cycles*folds))

# Create the arrays for the scores acquired with trialsRows voting process
precisionResultsTRows = np.empty((hyperparameterComb, cycles))
recallResultsTRows = np.empty((hyperparameterComb, cycles))
accuracyResultsTRows = np.empty((hyperparameterComb, cycles))

confMatricesResults = np.empty((cycles, hyperparameterComb, 2, 2))

# Dataframe for all the incorrectly predicted classes in all folds
foldsIncorrect = pd.DataFrame()

# Dataframe for all the trialRows feature rows
foldTrialResults = pd.DataFrame()

for i in range(cycles):

    print("\n\n########################## CYCLE: ", i)



    # ---------------------------------------------------
    print("# Doing a grid search on all the hyperparameter combinations given #\n")

    # Current hyperparametercombination, used for storing the results
    hComb = 0
    # Go through all the hyperparameter combinations
    for C in hyperParameters["C"]:
        for gamma in hyperParameters["gamma"]:
            print("## Kernel=rbf, C={}, gamma={}".format(C, gamma))

            # Create the stratifiedFold class here so that for each hyperparameter combination
            # the folds have the same randomisation
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1+i)

            # Empty the foldsIncorrect dataframe
            foldsIncorrect = foldsIncorrect.iloc[0:0]

            # Empty the foldTrialResults dataframe
            foldTrialResults = foldTrialResults.iloc[0:0]

            fold_n = 0
            # Go through all the splits
            for train_index, test_index in cv.split(X, y):
                fold_n += 1




                # Our classifier ( some documented runs had class_weight="balanced" )
                clf = svm.SVC(kernel="rbf", C=C, gamma=gamma)

                clf.fit(X.take(train_index), y[train_index])
                y_pred = clf.predict(X.take(test_index))

                # Store the values for each fold
                precisionResults[hComb][i*folds + (fold_n-1)] = precision_score(y[test_index], y_pred, pos_label=0)
                recallResults[hComb][i*folds + (fold_n-1)] = recall_score(y[test_index], y_pred, pos_label=0)
                accuracyResults[hComb][i*folds + (fold_n-1)] = accuracy_score(y[test_index], y_pred)

                # Nondyslectic results
                precisionResultsND[hComb][i*folds + (fold_n-1)] = precision_score(y[test_index], y_pred, pos_label=1)
                recallResultsND[hComb][i*folds + (fold_n-1)] = recall_score(y[test_index], y_pred, pos_label=1)

                foldDF = indexX.take(test_index)
                #print("fold", fold_n)

                foldDF["dyslexia"] = y[test_index]
                foldDF["predicted"] = y_pred

                if trialRows:
                    foldTrialResults = pd.concat([foldTrialResults, foldDF], axis=0).sort_index()
                    #print(foldTrialResults)

                incorrect = foldDF[foldDF["dyslexia"] != foldDF["predicted"]]
                #print("Incorrectly predicted: ", incorrect)

                foldsIncorrect = pd.concat([foldsIncorrect, incorrect], axis=0)
                #print(foldsIncorrect)













                # Calculate the confusion matrix
                cnf_matrix = confusion_matrix(y[test_index], y_pred)
                #print("\n", cnf_matrix)

                confMatricesFolds.append(cnf_matrix)

            # This adds the confusion matrices to our 4D array
            confMatricesResults[i, hComb, :, :] = np.sum(np.array(confMatricesFolds), axis=0)
            print("Current combination: ", hComb)

            foldsIncorrectList = foldsIncorrect.index.tolist()

            if trialRows:
                #print(foldTrialResults)
                currentPatient = foldTrialResults.index[0][:3]
                #print(currentPatient)
                # For trials on rows: remove the test subject ID
                foldsIncorrectList = [element[3:] for element in foldsIncorrectList]
                y_predVoted = list()
                y_trueVoted = list()
                #print(foldsIncorrectList)

                # Calculate predictions for each subject. Each trial gives a "vote" for the subject
                # being dyslectic or not. If the subject has at least 5 votes, then it is predicted to
                # have dyslexia.
                votes = 0
                overallVote = 0
                for row in foldTrialResults.itertuples():

                    id = row.Index[:3]

                    # If the patient has changed save results and get ready for new patient
                    if id > currentPatient:
                        # Add the vote results to our list
                        y_predVoted.append(overallVote)

                        # Make list with the true (actual) values of "dyslexia" for the patients
                        y_trueVoted.append(foldTrialResults.loc["{}T3".format(currentPatient)]["dyslexia"])

                        # Update the current patient
                        currentPatient = id

                        # Reset
                        overallVote = 0
                        votes = 0

                    # If the number of votes > 5 then the patient is deemed dyslectic
                    if votes > 5 and overallVote == 0:
                        overallVote = 1

                    # Add the vote given (1 or 0) to the total
                    votes += row.predicted

                    # If we are at the last row of the whole dataframe, save the data for the patient
                    if row.Index == foldTrialResults.tail(1).index.item():
                        #print("Last subject")
                        y_predVoted.append(overallVote)
                        y_trueVoted.append(foldTrialResults.loc["{}T3".format(currentPatient)]["dyslexia"])

                #print(y_predVoted, len(y_predVoted))
                #print(y_trueVoted)

                # Save the scores. The first two are for dyslectics only
                precisionResultsTRows[hComb][i] = precision_score(y_trueVoted, y_predVoted, pos_label=0)
                recallResultsTRows[hComb][i] = recall_score(y_trueVoted, y_predVoted, pos_label=0)
                print(recallResultsTRows[hComb][i])
                accuracyResultsTRows[hComb][i] = accuracy_score(y_trueVoted, y_predVoted)

            # If we are doing the first cycle append the list of all the incorrect predictions
            if i == 0:
                allIncorrect.append(foldsIncorrectList)
            else:
                # If we are past the first cycle extend the lists of each hyperparameter
                # combinations with their incorrect predictions
                allIncorrect[hComb].extend(foldsIncorrectList)




            # Move to the next hyperparameter combination
            hComb += 1

            # Empty the contents of confMatrices for the next folds
            confMatricesFolds.clear()
            #print("\nAverage precision for predicting dyslexia: %0.2f (+/- %0.2f)" % (foldPrecisions.mean(), foldPrecisions.std()))
            #print("\nAverage recall for predicting dyslexia: %0.2f (+/- %0.2f)" % (foldRecalls.mean(), foldRecalls.std()))

            #print("\n\nAverage precision for predicting not having dyslexia: %0.2f (+/- %0.2f)" % (foldPrecisionsND.mean(), foldPrecisionsND.std()))
            #print("\nAverage recall for predicting not having dyslexia: %0.2f (+/- %0.2f)" % (foldRecallsND.mean(), foldRecallsND.std()))


# Plot normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, normalize=True)
#print(confMatricesResults)
#recallResultsList = list(chain(*recallResults))

# Results for accuracy
accuracyResultsMean = accuracyResults.mean(axis=1)
accuracyResultsStd = accuracyResults.std(axis=1)

precisionResultsMean = precisionResults.mean(axis=1)
precisionResultsStd = precisionResults.std(axis=1)

precisionResultsNDMean = precisionResultsND.mean(axis=1)
precisionResultsNDStd = precisionResultsND.std(axis=1)

recallResultsMean = recallResults.mean(axis=1)
print(recallResultsMean)
recallResultsStd = recallResults.std(axis=1)

recallResultsNDMean = recallResultsND.mean(axis=1)
recallResultsNDStd = recallResultsND.std(axis=1)

confMatricesResultsMean = confMatricesResults.mean(axis=0)
#print(confMatricesResultsMean)

# Results for trialRows voting process
if trialRows:
    accuracyResultsTRowsMean = accuracyResultsTRows.mean(axis=1)
    accuracyResultsTRowsStd = accuracyResultsTRows.std(axis=1)

    precisionResultsTRowsMean = precisionResultsTRows.mean(axis=1)
    precisionResultsTRowsStd = precisionResultsTRows.std(axis=1)

    recallResultsTRowsMean = recallResultsTRows.mean(axis=1)
    recallResultsTRowsStd = recallResultsTRows.std(axis=1)
    print(recallResultsTRowsMean)

    print("\nResults for TrialRows voting: \n")
    j = 0
    for C in hyperParameters["C"]:
        for gamma in hyperParameters["gamma"]:
            print("#  C = {}, gamma = {}, mean recall = {:.3f}, mean Accuracy: {:.3f}".format(C,
                                                                                              gamma,
                                                                                              recallResultsTRowsMean[j],
                                                                                              accuracyResultsTRowsMean[j]))

            hyperparameterResultsTRows.append({"C": C, "gamma": gamma,
                                           "meanPrecision": precisionResultsTRowsMean[j],
                                           "meanRecall": recallResultsTRowsMean[j],
                                           "stdPrecisionError": precisionResultsTRowsStd[j],
                                           "stdRecallError": recallResultsTRowsStd[j],
                                           "meanAccuracy": accuracyResultsTRowsMean[j],
                                           "stdAccuracyError": accuracyResultsTRowsStd[j],
                                           "HyperparamCombNum": j})

            j += 1

    #print(hyperparameterResultsDF)

    # Print the summary of the results ###########################################################
    print("\n\n______________Top 10 results for trialRows voting______________\n")
    print("Cycles done: ", cycles)
    print("\n")

    sortedList = sorted(hyperparameterResultsTRows, key=itemgetter("meanRecall"), reverse=True)
    for row in sortedList[:50]:
        print("C: {}  gamma: {}  meanRecall: {:.3f} stdRecallError: {:.3f}  meanPrecision: {:.3f} stdPrecisionError: {:.3f} meanAccuracy: {:.3f} stdAccuracyError: {:.3f}".format(row["C"],
                                                                        row["gamma"],
                                                                        row["meanRecall"],
                                                                        row["stdRecallError"],
                                                                        row["meanPrecision"],
                                                                        row["stdPrecisionError"],
                                                                        row["meanAccuracy"],
                                                                        row["stdAccuracyError"]))
        #print("\n{}\n".format(row["confMatrix"]))

#print(allIncorrect)

# Create the final results table with mean values and the standard deviations
j = 0


print("Results for the search: \n")
for C in hyperParameters["C"]:
        for gamma in hyperParameters["gamma"]:
            print("#  C = {}, gamma = {}, mean recall = {:.3f}, meanRecallND: {:.3f}, mean Accuracy: {:.3f}".format(C,
                                                                                                                    gamma,
                                                                                                                    recallResultsMean[j],
                                                                                                                    recallResultsNDMean[j],
                                                                                                                    accuracyResultsMean[j]))

            hyperparameterResults.append({"C": C, "gamma": gamma,
                                           "meanPrecision": precisionResultsMean[j],
                                           "meanRecall": recallResultsMean[j],
                                           "stdPrecisionError": precisionResultsStd[j],
                                           "stdRecallError": recallResultsStd[j],
                                           "meanPrecisionND": precisionResultsNDMean[j],
                                           "meanRecallND": recallResultsNDMean[j],
                                           "stdPrecisionNDError": precisionResultsNDStd[j],
                                           "stdRecallNDError": recallResultsNDStd[j],
                                           "meanAccuracy": accuracyResultsMean[j],
                                           "stdAccuracyError": accuracyResultsStd[j],
                                           "HyperparamCombNum": j,
                                           "confMatrix": confMatricesResultsMean[j]})

            j += 1

#print(hyperparameterResultsDF)

# Calculate the mean and variance of the results



# Print the summary of the results ###########################################################
print("\n\n______________Top 10 results______________\n")
print("Cycles done: ", cycles)
print("\n")

sortedList = sorted(hyperparameterResults, key=itemgetter("meanRecall"), reverse=True)
for row in sortedList[:50]:
    print("C: {}  gamma: {}  meanRecall: {:.3f} stdRecallError: {:.3f}  meanPrecision: {:.3f} stdPrecisionError: {:.3f} meanRecallND: {:.3f} stdRecallNDError: {:.3f}  meanPrecisionND: {:.3f} stdPrecisionNDError: {:.3f} meanAccuracy: {:.3f} stdAccuracyError: {:.3f}".format(row["C"],
                                                                       row["gamma"],
                                                                       row["meanRecall"],
                                                                       row["stdRecallError"],
                                                                       row["meanPrecision"],
                                                                       row["stdPrecisionError"],
                                                                       row["meanRecallND"],
                                                                       row["stdRecallNDError"],
                                                                       row["meanPrecisionND"],
                                                                       row["stdPrecisionNDError"],
                                                                       row["meanAccuracy"],
                                                                       row["stdAccuracyError"]))
    print("\n{}\n".format(row["confMatrix"]))


counter = collections.Counter(allIncorrect[sortedList[0]["HyperparamCombNum"]])
print("The most common incorrectly predicted for C={} gamma={} :{} ".format(sortedList[0]["C"], sortedList[0]["gamma"], counter.most_common(20)))

# Calculate the mean confusion matrix of all the confusion matrices
#meanConfMatrix = np.mean(np.array(confMatrices), axis=0)
#print("\n The mean confusion matrix: \n", meanConfMatrix)

#print("\nAverage precision for predicting dyslexia: %0.2f (+/- %0.2f)" % (cyclePrecisions.mean(), cyclePrecisions.std() * 2))
#print("\nAverage recall for predicting dyslexia: %0.2f (+/- %0.2f)" % (cycleRecalls.mean(), cycleRecalls.std() * 2))

#print("\n\nAverage precision for predicting not having dyslexia: %0.2f (+/- %0.2f)" % (cyclePrecisionsND.mean(), cyclePrecisionsND.std() * 2))
#print("\nAverage recall for predicting not having dyslexia: %0.2f (+/- %0.2f)" % (cycleRecallsND.mean(), cycleRecallsND.std() * 2))
#print([a[1] for a in counter.most_common()])

print("\nTotal time:")
print("\nDone in %0.3fs" % (time() - t0))

plt.hist([a[1] for a in counter.most_common()])
plt.show()

plt.plot(recallResultsMean, "k", recallResultsMean, "bo")
plt.ylabel("Recall")
plt.xlabel("Hyperparameter combination")
plt.title("Cycles:{} folds:{}".format(cycles, folds))
#plt.show()

# Create the histograms
plt.figure(1)
plt.suptitle("Recall score histograms of folds, cycles:{} folds:{}".format(cycles, folds))

for i in range(hyperparameterComb):

    plt.subplot(len(hyperParameters["C"]), len(hyperParameters["gamma"]), i+1)
    plt.hist(recallResults[i])

#plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.subplots_adjust(top=0.95, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                    wspace=0.25)

