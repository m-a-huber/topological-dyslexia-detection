# Peter Raatikainen 2018
# This script gives a result of different hyperparameter combinations and their scores
# by using a own gridsearch function with cross-validation

from time import time
import argparse
import itertools
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import matplotlib.pyplot as plt
import collections # This is for Counter
from operator import itemgetter # This is for sorting the hyperparameter results
from itertools import chain
from matplotlib.ticker import NullFormatter

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve

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
topF = 35

# Load the data from the file
df = pd.read_csv(args.open, sep=';', header=0, index_col="id", decimal=',')












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

# List of the features
features = list(X)

# Find out the best parameters for training the RF
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

# This list contains all of the feature importances evaluated by RandomForest
allImportances = list()

# List for storing all of the confusion matrices
confMatricesFolds = list()

# List for feature importances
featureImportances = list()

# List for storing the best parameters from each cycle'
bestParams = list()
# Original grid stored here for historical detail (used while developing method):
# (Observe that in fast-paced research project we reused names "C", "gamma" for RF parameters)
# (It is apparent from the code below that they were in fact C==max_features, gamma==n_estimators)
# hyperParameters = {"C": [2, 3, 4, 5, 6, 8, 10, 15, 20],
#                    "gamma": [5, 20, 30, 50, 80, 100, 300, 500, 1000, 2000]}

# Final documented results were computed by fine-tuning parameters here,
# after optimal range had been narrowed down using the grid search:
hyperParameters = {"C": [550],
                  "gamma": [20]}

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

    #outerCV = StratifiedKFold(n_splits=folds, shuffle=True, random_state=i+1)

    # ---------------------------------------------------
    print("# Doing a grid search on all the hyperparameter combinations given #\n")

    # Current hyperparametercombination, used for storing the results
    hComb = 0
    # Go through all the hyperparameter combinations
    for C in hyperParameters["C"]: #C=Max_features
        for gamma in hyperParameters["gamma"]: #gamma = n_estimators
            print("## RandomForest, C={}, gamma={}".format(C, gamma))

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
                #print("\nFold: ", fold_n)
                #print(test_index)
                #print(X.take(train_index))

                # Our classifier ( some documented runs had class_weight="balanced" )
                clf = RandomForestClassifier(max_depth=None, max_features=C, n_estimators=gamma, random_state=1)

                clf.fit(X.take(train_index), y[train_index])
                y_pred = clf.predict(X.take(test_index))

                # Store the values for each fold
                precisionResults[hComb][i*folds + (fold_n-1)] = precision_score(y[test_index], y_pred, pos_label=0)
                recallResults[hComb][i*folds + (fold_n-1)] = recall_score(y[test_index], y_pred, pos_label=0)
                accuracyResults[hComb][i*folds + (fold_n-1)] = accuracy_score(y[test_index], y_pred)

                # Nondyslectic results
                precisionResultsND[hComb][i*folds + (fold_n-1)] = precision_score(y[test_index], y_pred, pos_label=1)
                recallResultsND[hComb][i*folds + (fold_n-1)] = recall_score(y[test_index], y_pred, pos_label=1)

                foldDF = X.take(test_index)
                #print("fold", fold_n)
                #print(foldDF)
                foldDF["dyslexia"] = y[test_index]
                foldDF["predicted"] = y_pred

                if trialRows:
                    foldTrialResults = pd.concat([foldTrialResults, foldDF], axis=0).sort_index()
                    #print(foldTrialResults)

                incorrect = foldDF[foldDF["dyslexia"] != foldDF["predicted"]]
                #print("Incorrectly predicted: ", incorrect)

                foldsIncorrect = pd.concat([foldsIncorrect, incorrect], axis=0)

                # Save the n (=10) most important features
                topImportances = sorted(list(zip(features, clf.feature_importances_)), key=itemgetter(1), reverse=True)[:10]
                topImportances = [j[0] for j in topImportances]
                # If we are doing the first hyperparam cycle append the list of all the feature importances
                if i == 0 and fold_n == 1:
                    #print(topImportances)
                    allImportances.append(topImportances)
                else:
                    # If we are past the first cycle extend the lists of each hyperparameter
                    # combinations with their feature importances
                    #print(topImportances)
                    allImportances[hComb].extend(topImportances)

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

            # Save featureImportances
            featureImportances = clf.feature_importances_

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
            print("#  Max_features = {}, n_estimators = {}, mean recall = {:.3f}, meanRecallND: {:.3f}, mean Accuracy: {:.3f}".format(C,
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

sortedList = sorted(hyperparameterResults, key=itemgetter("meanRecall", "meanAccuracy"), reverse=True)
for row in sortedList[:40]:
    print("Max_features: {}  n_estimators: {}  meanRecall: {:.3f} stdRecallError: {:.3f}  meanPrecision: {:.3f} stdPrecisionError: {:.3f} meanRecallND: {:.3f} stdRecallNDError: {:.3f}  meanPrecisionND: {:.3f} stdPrecisionNDError: {:.3f} meanAccuracy: {:.3f} stdAccuracyError: {:.3f}".format(row["C"],
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

# Show the most common incorrectly predicted for the best result
counter = collections.Counter(allIncorrect[sortedList[0]["HyperparamCombNum"]])
print("The most common incorrectly predicted for C={} gamma={} :{} ".format(sortedList[0]["C"], sortedList[0]["gamma"], counter.most_common(20)))

# Calculate the mean confusion matrix of all the confusion matrices
#meanConfMatrix = np.mean(np.array(confMatrices), axis=0)
#print("\n The mean confusion matrix: \n", meanConfMatrix)

#print("\nAverage precision for predicting dyslexia: %0.2f (+/- %0.2f)" % (cyclePrecisions.mean(), cyclePrecisions.std() * 2))
#print("\nAverage recall for predicting dyslexia: %0.2f (+/- %0.2f)" % (cycleRecalls.mean(), cycleRecalls.std() * 2))

#print("\n\nAverage precision for predicting not having dyslexia: %0.2f (+/- %0.2f)" % (cyclePrecisionsND.mean(), cyclePrecisionsND.std() * 2))
#print("\nAverage recall for predicting not having dyslexia: %0.2f (+/- %0.2f)" % (cycleRecallsND.mean(), cycleRecallsND.std() * 2))


print("\nTotal time:")
print("\nDone in %0.3fs" % (time() - t0))

print(allImportances[sortedList[0]["HyperparamCombNum"]])


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
#plt.show()



# Count occurences of trials specific features
data_items = allImportances[sortedList[0]["HyperparamCombNum"]]
search = ["T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11"]
resultTrials = {k:0 for k in search}
for item in data_items:
    for search_item in search:
        if search_item in item:
            resultTrials[search_item]+=1
print(resultTrials)

# Count all occurences of sentence specific features #########################
# Remove the first two characters of each name of the features. E.g. T3FFixDurBin3 -> FFixDurBin3
bestModelSentence = [element[2:] for element in data_items] #if element[0] == "0" or element[0] == "1"
#print(bestModelSentence)

# Create a list of the feature names with the sentence label as the first character
bestModelSentenceData1 = [element[0] for element in bestModelSentence if element[0] != '0' and element[0] != '1']
#print(bestModelSentenceData1)

# Create a list of sentence labels of the rest of the feature names, i.e., the ones starting with either 0 or 1
bestModelSentenceData2 = [element[1] for element in bestModelSentence if element[0] == '0' or element[0] == '1']
#print(bestModelSentenceData2)

# Combine the above and show results.
bestModelSentenceData1.extend(bestModelSentenceData2)
print("The sentences: ", bestModelSentenceData1)

# Show the frequencies of each sentence related feature
counter = collections.Counter(bestModelSentenceData1)
print("Frequencies of each sentence related feature: ", counter.most_common(5))

# Show feature importance from RandomForestClassifier
counter = collections.Counter(allImportances[sortedList[0]["HyperparamCombNum"]])
print("Feature importance for max_features={} n_estimators={} :{} ".format(sortedList[0]["C"], sortedList[0]["gamma"], counter.most_common(topF)))
features = list(X)
for i in range(len(featureImportances)):
    print("{}: {}".format(features[i], featureImportances[i]))

# Create ROC curve
#roc_curve


# Show a chart with Recall vs Accuracy
plt.plot(recallResultsMean, "k", accuracyResultsMean, "bo")
plt.ylabel("Recall")
plt.xlabel("Accuracy")
plt.title("Cycles:{} folds:{}".format(cycles, folds))
#plt.show()


#############################################################################################
# Save the most frequent features measured by importance to csv file
topFeatures = [i[0] for i in counter.most_common(15)]
print(topFeatures)
topFeatures.append("dyslexia")
chosen = df[topFeatures]

chosen.to_csv("featuresOwn1test.csv", sep=";", index=True, index_label="id", decimal=',')
print("...featuresOwn written to csv file.")
#############################################################################################
