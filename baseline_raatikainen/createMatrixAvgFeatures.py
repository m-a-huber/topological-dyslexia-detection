# Peter Raatikainen 2018
# This script creates the features in a matrix from our whole data. The matrix hold data on
# how many times the patients point of interest area has shifted between the sentences. It also
# holds information on the amount of fixations and the amplitude of the saccades in each sentence.
# for the chosen machine learning method

import numpy as np
from sklearn import preprocessing, model_selection
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.preprocessing import MinMaxScaler

# This is for plotting the data
from pandas.tools import plotting
import seaborn as sns
import argparse

# The Argument parser and arguments -----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--nonorm", help="Turn off normalisation of the features", action="store_true")
args = parser.parse_args()
# ---------------------------------------------------------------------------

################################################################
# List of global variables used for controlling the output
makeSumMatrix = 1
histograms = True
trialRows = False
normValue = ""
normalisation = True
zpafBorder = "10"
if normalisation:
    normValue = "Norm"

################################################################


# List of variables/columns wanted
# All: id, trialid, TRIAL_INDEX, CURRENT_FIX_DURATION, CURRENT_FIX_INDEX, CURRENT_FIX_X, CURRENT_FIX_Y,
#      PREVIOUS_SAC_DURATION, PREVIOUS_SAC_AMPLITUDE, CURRENT_FIX_INTEREST_AREA_INDEX, IA_group_label,
#      ZPAF_10_percentile_Group, CURRENT_FIX_START, IA_grouped
variableList = ["id",
                "trialid",
                "IA_group_label",
                "CURRENT_FIX_DURATION",
                "PREVIOUS_SAC_DURATION",
                "PREVIOUS_SAC_AMPLITUDE",
                "ZPAF_{}_percentile_Group".format(zpafBorder)]

# Load the data from the file
# Create a dataframe with the wanted variables to be used for creating the features
variables = pd.read_csv('fixations_corrected_task1.csv',
                        sep=';',
                        header=0,
                        decimal=',',
                        usecols=variableList,
                        na_values={"IA_group_label": " ",
                        "PREVIOUS_SAC_AMPLITUDE": " ",
                        "PREVIOUS_SAC_DURATION": " "},
                        low_memory=False)

# Print the data types of the columns
print(variables.dtypes)

# Remove whitespace cells and replace them with 0
variables = variables.replace(' ', 0)

# Replace commas from the numbers into dots for the numeric conversion
# The conversion won't work without this
#variables = variables.replace(',', '.')
#print(variables)

#variables.to_csv("featuresTest.csv", sep=";", decimal=',')

# Iterate over all the rows and create the features dataframe ###########################

# First assign the ID of the first test subject
currentPatient = variables.loc[0]["id"]

# The first patients dyslexia value
currentPatientDyslexia = variables.loc[0]["ZPAF_{}_percentile_Group".format(zpafBorder)]

# The patients amount of fixations that did not hit the sentences areas
currentPatientOutFixSum = 0

# The name of the first interest area aka sentence
previousIA = variables.loc[0]["IA_group_label"]

# The number of the first trial
previousTrial = variables.loc[0]["trialid"]

# Create the columns for the features dataframe --------------------------------

# The letters representing each sentence
sentences = ["F", "T", "D", "L"]

# More columns will be added after these
featuresColumns = ["dyslexia"]

# This list is for the quantile values of Fixation duration
fixDurQuantile = [147, 178, 214, 272]

# This list is for the quantile values of Saccades duration
sacDurQuantile = [20, 24, 29, 41]

# This list is for the quantile values of Saccade amplitude = distance
sacAmpQuantile = [1.23, 1.94, 2.72, 4.08]

if makeSumMatrix == 2:

    # 4x6 matrix
    transMAvg = np.zeros((4, 6))
    print(transMAvg)

    for s in sentences:

        for e in sentences:
            featuresColumns.append("Sum{}-{}".format(s,e))

        featuresColumns.append("{}FixSum".format(s))
        featuresColumns.append("{}SaccAmplSum".format(s))

    transM = np.zeros((10, 4, 6))
    print(transM)

elif makeSumMatrix == 1:

    if histograms:
        # 10x4x18 matrix used for storing the transition matrix, and on the diagonal of this the
        # histograms with 5 bins for amount of fixations, saccade amplitudes and saccade length
        transM = np.zeros((10, 4, 19))

    else:
        # 10x4x6 matrix used for storing the transition times between each area of interest group = sentence
        # Making this bigger now to fit more data. First the sum of fixations in each sentence.
        # Also saving the sum of the amplitudes in each sentence
        transM = np.zeros((10, 4, 6))

    print(transM)

    for trial in range(2,12):

        for s in sentences:

            for e in sentences:

                featuresColumns.append("T{}{}-{}".format(trial,s,e))

            if histograms:

                for e in range(1,6):
                    featuresColumns.append("T{}{}FixDurBin{}".format(trial,s,e))

                for e in range(1,6):
                    featuresColumns.append("T{}{}SaccDurBin{}".format(trial,s,e))

                for e in range(1,6):
                    featuresColumns.append("T{}{}SaccAmpBin{}".format(trial,s,e))

            else:

                featuresColumns.append("T{}{}FixSum".format(trial,s))
                featuresColumns.append("T{}{}SaccAmplSum".format(trial,s))

elif makeSumMatrix == 3:

    # 10x4x6 matrix used for storing the transition times between each area of interest group = sentence
    # Making this bigger now to fit more data. First the sum of fixations in each sentence.
    # Also saving the sum of the amplitudes in each sentence
    transM = np.zeros((10, 4, 6))
    print(transM)

    for s in sentences:

        for e in sentences:
            featuresColumns.append("{}-{}".format(s,e))

        featuresColumns.append("{}FixSum".format(s))
        featuresColumns.append("{}SaccAmplSum".format(s))

elif makeSumMatrix == 4:

    print(transM)

    for trial in range(2,12):

        for s in sentences:

            for e in sentences:

                featuresColumns.append("T{}{}-{}".format(trial,s,e))

print(featuresColumns)

# -------------------------------------------------------------------------------

# Create the dataframe columns
features = pd.DataFrame(columns=featuresColumns)

# Go through the whole data and collect the data of the transitions into the transM matrix
#/////////////////////////////////////////////////////////////////////////////////////////
for row in variables.itertuples():

    ### If we want all the trials mean values into one matrix ####################
    if makeSumMatrix == 2:

        # If the ID of the test subject changes save data to dataframe and prepare for new patient
        if row.id > currentPatient and row.Index != len(variables) - 1:

            # Save the collected data into the features dataframe --------------------
            transMAvg = np.mean(transM, axis=0)
            #print(transMAvg)
            chainedListFinal = list(chain(*transMAvg))

            # Insert to the start of the list values to do with the patient
            # Amount of fixations outside the sentences 
            #chainedListFinal.insert(0,currentPatientOutFixSum)
            # ZPAF_125_percentile_Group value
            chainedListFinal.insert(0,currentPatientDyslexia)

            features.loc[currentPatient] = chainedListFinal
            # ------------------------------------------------------------------------
            # RESET VALUES

            currentPatientOutFixSum = 0 # Zero for the next patient
            # Empty the transM matrix for the next patients data
            transM.fill(0)

            # Update current patient and their dyslexia value
            currentPatient = row.id
            currentPatientDyslexia = row[7] # ZPAF value

            # Also correct the previous interest area because the patient changed!
            previousIA = row.IA_group_label

        # If the trial has not changed. This is used to avoid recording wrong information about transitions
        # between the trials.
        if row.trialid == previousTrial:
            # If the current sentence is different than the previous
            if row.IA_group_label != previousIA:

                # The variable for placing the trial details in the 3d matrix
                matrixTrial = row.trialid - 2

                if row.IA_group_label == "first":

                    if previousIA == "task":
                        transM[matrixTrial][1][0] += 1
                    elif previousIA == "distr":
                        transM[matrixTrial][2][0] += 1
                    elif previousIA == "last":
                        transM[matrixTrial][3][0] += 1

                if row.IA_group_label == "task":

                    if previousIA == "first":
                        transM[matrixTrial][0][1] += 1
                    elif previousIA == "distr":
                        transM[matrixTrial][2][1] += 1
                    elif previousIA == "last":
                        transM[matrixTrial][3][1] += 1

                if row.IA_group_label == "distr":

                    if previousIA == "first":
                        transM[matrixTrial][0][2] += 1
                    elif previousIA == "task":
                        transM[matrixTrial][1][2] += 1
                    elif previousIA == "last":
                        transM[matrixTrial][3][2] += 1

                if row.IA_group_label == "last":

                    if previousIA == "first":
                        transM[matrixTrial][0][3] += 1
                    elif previousIA == "task":
                        transM[matrixTrial][1][3] += 1
                    elif previousIA == "distr":
                        transM[matrixTrial][2][3] += 1

                # If the label is empty meaning the fixation went out of the sentences
                if pd.isnull(row.IA_group_label):

                    currentPatientOutFixSum += 1

                previousIA = row.IA_group_label

            # If the current sentence is the same as before
            elif row.IA_group_label == previousIA:

                # The variable for placing the trial details in the 3d matrix
                matrixTrial = row.trialid - 2

                if row.IA_group_label == "first":
                    transM[matrixTrial][0][0] += 1
                    # Increase the sum of the fixation durations in the sentence
                    transM[matrixTrial][0][4] += row.CURRENT_FIX_DURATION
                    # Increase the sum of the saccade amplitudes in the sentence
                    transM[matrixTrial][0][5] += row.PREVIOUS_SAC_AMPLITUDE

                elif row.IA_group_label == "task":
                    transM[matrixTrial][1][1] += 1

                    # Increase the sum of the fixation durations in the sentence
                    transM[matrixTrial][1][4] += row.CURRENT_FIX_DURATION
                    # Increase the sum of the saccade amplitudes in the sentence
                    transM[matrixTrial][1][5] += row.PREVIOUS_SAC_AMPLITUDE

                elif row.IA_group_label == "distr":
                    transM[matrixTrial][2][2] += 1

                    # Increase the sum of the fixation durations in the sentence
                    transM[matrixTrial][2][4] += row.CURRENT_FIX_DURATION
                    # Increase the sum of the saccade amplitudes in the sentence
                    transM[matrixTrial][2][5] += row.PREVIOUS_SAC_AMPLITUDE

                elif row.IA_group_label == "last":
                    transM[matrixTrial][3][3] += 1

                    # Increase the sum of the fixation durations in the sentence
                    transM[matrixTrial][3][4] += row.CURRENT_FIX_DURATION
                    # Increase the sum of the saccade amplitudes in the sentence
                    transM[matrixTrial][3][5] += row.PREVIOUS_SAC_AMPLITUDE
        else:

            # Update that the trial has changed
            previousTrial = row.trialid

            # Also update the previousIA because the trial has changed
            previousIA = row.IA_group_label

        # If we are at the last row of the dataframe save the results for the last patient
        if row.Index == len(variables) - 1:

            print("last element", row)
            # Save the collected data into the features dataframe --------------------
            transMAvg = np.mean(transM, axis=0)
            chainedListFinal = list(chain(*transMAvg))

            # Insert to the start of the list values to do with the patient
            # Amount of fixations outside the sentences
            #chainedListFinal.insert(0,currentPatientOutFixSum)
            # ZPAF_125_percentile_Group value
            chainedListFinal.insert(0,currentPatientDyslexia)

            features.loc[currentPatient] = chainedListFinal
            # ------------------------------------------------------------------------

    #### If we want the trials in separate matrices #############
    elif makeSumMatrix == 1:


        # If the ID of the test subject changes, save data to dataframe and prepare for new patient
        if row.id > currentPatient and row.Index != len(variables) - 1:

            # Save the collected data into the features dataframe --------------------
            chainedList1 = list(chain(*transM))
            chainedListFinal = list(chain(*chainedList1))

            # Insert to the start of the list values to do with the patient
            # Amount of fixations outside the sentences
            #chainedListFinal.insert(0,currentPatientOutFixSum)
            # ZPAF_125_percentile_Group value
            chainedListFinal.insert(0,currentPatientDyslexia)

            features.loc[currentPatient] = chainedListFinal
            # ------------------------------------------------------------------------
            # RESET VALUES

            currentPatientOutFixSum = 0 # Zero for the next patient
            # Empty the transM matrix for the next patients data
            transM.fill(0)

            # Update current patient and their dyslexia value
            currentPatient = row.id
            currentPatientDyslexia = row[7] #ZPAF_percentile_Group

            # Also correct the previous interest area because the patient changed!
            previousIA = row.IA_group_label


        # If the trial has not changed. This is used to avoid recording wrong information about transitions
        # between the trials.
        if row.trialid == previousTrial:

            # If the current sentence is different than the previous
            if row.IA_group_label != previousIA:

                # The variable for placing the trial details in the 3d matrix
                matrixTrial = row.trialid - 2

                if row.IA_group_label == "first":

                    if previousIA == "task":
                        transM[matrixTrial][1][0] += 1
                    elif previousIA == "distr":
                        transM[matrixTrial][2][0] += 1
                    elif previousIA == "last":
                        transM[matrixTrial][3][0] += 1

                if row.IA_group_label == "task":

                    if previousIA == "first":
                        transM[matrixTrial][0][1] += 1
                    elif previousIA == "distr":
                        transM[matrixTrial][2][1] += 1
                    elif previousIA == "last":
                        transM[matrixTrial][3][1] += 1

                if row.IA_group_label == "distr":

                    if previousIA == "first":
                        transM[matrixTrial][0][2] += 1
                    elif previousIA == "task":
                        transM[matrixTrial][1][2] += 1
                    elif previousIA == "last":
                        transM[matrixTrial][3][2] += 1

                if row.IA_group_label == "last":

                    if previousIA == "first":
                        transM[matrixTrial][0][3] += 1
                    elif previousIA == "task":
                        transM[matrixTrial][1][3] += 1
                    elif previousIA == "distr":
                        transM[matrixTrial][2][3] += 1

                # If the label is empty meaning the fixation went out of the sentences
                if pd.isnull(row.IA_group_label):

                    currentPatientOutFixSum += 1

                previousIA = row.IA_group_label

            # If the current sentence is the same as before
            elif row.IA_group_label == previousIA:

                # The variable for placing the trial details in the 3d matrix
                matrixTrial = row.trialid - 2

                if row.IA_group_label == "first":

                    transM[matrixTrial][0][0] += 1

                    # If we are making the histogram transition matrix
                    if histograms:

                        # Placing the value of the fixation duration into the correct bin
                        if row.CURRENT_FIX_DURATION <= fixDurQuantile[0]:
                            transM[matrixTrial][0][4] += 1
                        elif row.CURRENT_FIX_DURATION <= fixDurQuantile[1]:
                            transM[matrixTrial][0][5] += 1
                        elif row.CURRENT_FIX_DURATION <= fixDurQuantile[2]:
                            transM[matrixTrial][0][6] += 1
                        elif row.CURRENT_FIX_DURATION <= fixDurQuantile[3]:
                            transM[matrixTrial][0][7] += 1
                        elif row.CURRENT_FIX_DURATION > fixDurQuantile[3]:
                            transM[matrixTrial][0][8] += 1

                        # Placing the value of the saccade duration into the correct bin
                        if row.PREVIOUS_SAC_DURATION <= sacDurQuantile[0]:
                            transM[matrixTrial][0][9] += 1
                        elif row.PREVIOUS_SAC_DURATION <= sacDurQuantile[1]:
                            transM[matrixTrial][0][10] += 1
                        elif row.PREVIOUS_SAC_DURATION <= sacDurQuantile[2]:
                            transM[matrixTrial][0][11] += 1
                        elif row.PREVIOUS_SAC_DURATION <= sacDurQuantile[3]:
                            transM[matrixTrial][0][12] += 1
                        elif row.PREVIOUS_SAC_DURATION > sacDurQuantile[3]:
                            transM[matrixTrial][0][13] += 1

                        # Placing the value of the saccade amplitude into the correct bin
                        if row.PREVIOUS_SAC_AMPLITUDE <= sacAmpQuantile[0]:
                            transM[matrixTrial][0][14] += 1
                        elif row.PREVIOUS_SAC_AMPLITUDE <= sacAmpQuantile[1]:
                            transM[matrixTrial][0][15] += 1
                        elif row.PREVIOUS_SAC_AMPLITUDE <= sacAmpQuantile[2]:
                            transM[matrixTrial][0][16] += 1
                        elif row.PREVIOUS_SAC_AMPLITUDE <= sacAmpQuantile[3]:
                            transM[matrixTrial][0][17] += 1
                        elif row.PREVIOUS_SAC_AMPLITUDE > sacAmpQuantile[3]:
                            transM[matrixTrial][0][18] += 1

                    # Else we are making the normal matrix
                    else:
                        # Increase the sum of the fixation durations in the sentence
                        transM[matrixTrial][0][4] += row.CURRENT_FIX_DURATION
                        # Increase the sum of the saccade amplitudes in the sentence
                        transM[matrixTrial][0][5] += row.PREVIOUS_SAC_AMPLITUDE

                elif row.IA_group_label == "task":
                    transM[matrixTrial][1][1] += 1

                    # If we are making the histogram transition matrix
                    if histograms:

                        # Placing the value of the fixation duration into the correct bin
                        if row.CURRENT_FIX_DURATION <= fixDurQuantile[0]:
                            transM[matrixTrial][1][4] += 1
                        elif row.CURRENT_FIX_DURATION <= fixDurQuantile[1]:
                            transM[matrixTrial][1][5] += 1
                        elif row.CURRENT_FIX_DURATION <= fixDurQuantile[2]:
                            transM[matrixTrial][1][6] += 1
                        elif row.CURRENT_FIX_DURATION <= fixDurQuantile[3]:
                            transM[matrixTrial][1][7] += 1
                        elif row.CURRENT_FIX_DURATION > fixDurQuantile[3]:
                            transM[matrixTrial][1][8] += 1

                        # Placing the value of the saccade duration into the correct bin
                        if row.PREVIOUS_SAC_DURATION <= sacDurQuantile[0]:
                            transM[matrixTrial][1][9] += 1
                        elif row.PREVIOUS_SAC_DURATION <= sacDurQuantile[1]:
                            transM[matrixTrial][1][10] += 1
                        elif row.PREVIOUS_SAC_DURATION <= sacDurQuantile[2]:
                            transM[matrixTrial][1][11] += 1
                        elif row.PREVIOUS_SAC_DURATION <= sacDurQuantile[3]:
                            transM[matrixTrial][1][12] += 1
                        elif row.PREVIOUS_SAC_DURATION > sacDurQuantile[3]:
                            transM[matrixTrial][1][13] += 1

                        # Placing the value of the saccade amplitude into the correct bin
                        if row.PREVIOUS_SAC_AMPLITUDE <= sacAmpQuantile[0]:
                            transM[matrixTrial][1][14] += 1
                        elif row.PREVIOUS_SAC_AMPLITUDE <= sacAmpQuantile[1]:
                            transM[matrixTrial][1][15] += 1
                        elif row.PREVIOUS_SAC_AMPLITUDE <= sacAmpQuantile[2]:
                            transM[matrixTrial][1][16] += 1
                        elif row.PREVIOUS_SAC_AMPLITUDE <= sacAmpQuantile[3]:
                            transM[matrixTrial][1][17] += 1
                        elif row.PREVIOUS_SAC_AMPLITUDE > sacAmpQuantile[3]:
                            transM[matrixTrial][1][18] += 1

                    # Else we are making the normal matrix
                    else:
                        # Increase the sum of the fixation durations in the sentence
                        transM[matrixTrial][1][4] += row.CURRENT_FIX_DURATION
                        # Increase the sum of the saccade amplitudes in the sentence
                        transM[matrixTrial][1][5] += row.PREVIOUS_SAC_AMPLITUDE

                elif row.IA_group_label == "distr":
                    transM[matrixTrial][2][2] += 1

                    # If we are making the histogram transition matrix
                    if histograms:

                        # Placing the value of the fixation duration into the correct bin
                        if row.CURRENT_FIX_DURATION <= fixDurQuantile[0]:
                            transM[matrixTrial][2][4] += 1
                        elif row.CURRENT_FIX_DURATION <= fixDurQuantile[1]:
                            transM[matrixTrial][2][5] += 1
                        elif row.CURRENT_FIX_DURATION <= fixDurQuantile[2]:
                            transM[matrixTrial][2][6] += 1
                        elif row.CURRENT_FIX_DURATION <= fixDurQuantile[3]:
                            transM[matrixTrial][2][7] += 1
                        elif row.CURRENT_FIX_DURATION > fixDurQuantile[3]:
                            transM[matrixTrial][2][8] += 1

                        # Placing the value of the saccade duration into the correct bin
                        if row.PREVIOUS_SAC_DURATION <= sacDurQuantile[0]:
                            transM[matrixTrial][2][9] += 1
                        elif row.PREVIOUS_SAC_DURATION <= sacDurQuantile[1]:
                            transM[matrixTrial][2][10] += 1
                        elif row.PREVIOUS_SAC_DURATION <= sacDurQuantile[2]:
                            transM[matrixTrial][2][11] += 1
                        elif row.PREVIOUS_SAC_DURATION <= sacDurQuantile[3]:
                            transM[matrixTrial][2][12] += 1
                        elif row.PREVIOUS_SAC_DURATION > sacDurQuantile[3]:
                            transM[matrixTrial][2][13] += 1

                        # Placing the value of the saccade amplitude into the correct bin
                        if row.PREVIOUS_SAC_AMPLITUDE <= sacAmpQuantile[0]:
                            transM[matrixTrial][2][14] += 1
                        elif row.PREVIOUS_SAC_AMPLITUDE <= sacAmpQuantile[1]:
                            transM[matrixTrial][2][15] += 1
                        elif row.PREVIOUS_SAC_AMPLITUDE <= sacAmpQuantile[2]:
                            transM[matrixTrial][2][16] += 1
                        elif row.PREVIOUS_SAC_AMPLITUDE <= sacAmpQuantile[3]:
                            transM[matrixTrial][2][17] += 1
                        elif row.PREVIOUS_SAC_AMPLITUDE > sacAmpQuantile[3]:
                            transM[matrixTrial][2][18] += 1

                    # Else we are making the normal matrix
                    else:
                        # Increase the sum of the fixation durations in the sentence
                        transM[matrixTrial][2][4] += row.CURRENT_FIX_DURATION
                        # Increase the sum of the saccade amplitudes in the sentence
                        transM[matrixTrial][2][5] += row.PREVIOUS_SAC_AMPLITUDE

                elif row.IA_group_label == "last":
                    transM[matrixTrial][3][3] += 1

                    # If we are making the histogram transition matrix
                    if histograms:

                        # Placing the value of the fixation duration into the correct bin
                        if row.CURRENT_FIX_DURATION <= fixDurQuantile[0]:
                            transM[matrixTrial][3][4] += 1
                        elif row.CURRENT_FIX_DURATION <= fixDurQuantile[1]:
                            transM[matrixTrial][3][5] += 1
                        elif row.CURRENT_FIX_DURATION <= fixDurQuantile[2]:
                            transM[matrixTrial][3][6] += 1
                        elif row.CURRENT_FIX_DURATION <= fixDurQuantile[3]:
                            transM[matrixTrial][3][7] += 1
                        elif row.CURRENT_FIX_DURATION > fixDurQuantile[3]:
                            transM[matrixTrial][3][8] += 1

                        # Placing the value of the saccade duration into the correct bin
                        if row.PREVIOUS_SAC_DURATION <= sacDurQuantile[0]:
                            transM[matrixTrial][3][9] += 1
                        elif row.PREVIOUS_SAC_DURATION <= sacDurQuantile[1]:
                            transM[matrixTrial][3][10] += 1
                        elif row.PREVIOUS_SAC_DURATION <= sacDurQuantile[2]:
                            transM[matrixTrial][3][11] += 1
                        elif row.PREVIOUS_SAC_DURATION <= sacDurQuantile[3]:
                            transM[matrixTrial][3][12] += 1
                        elif row.PREVIOUS_SAC_DURATION > sacDurQuantile[3]:
                            transM[matrixTrial][3][13] += 1

                        # Placing the value of the saccade amplitude into the correct bin
                        if row.PREVIOUS_SAC_AMPLITUDE <= sacAmpQuantile[0]:
                            transM[matrixTrial][3][14] += 1
                        elif row.PREVIOUS_SAC_AMPLITUDE <= sacAmpQuantile[1]:
                            transM[matrixTrial][3][15] += 1
                        elif row.PREVIOUS_SAC_AMPLITUDE <= sacAmpQuantile[2]:
                            transM[matrixTrial][3][16] += 1
                        elif row.PREVIOUS_SAC_AMPLITUDE <= sacAmpQuantile[3]:
                            transM[matrixTrial][3][17] += 1
                        elif row.PREVIOUS_SAC_AMPLITUDE > sacAmpQuantile[3]:
                            transM[matrixTrial][3][18] += 1

                    # Else we are making the normal matrix
                    else:
                        # Increase the sum of the fixation durations in the sentence
                        transM[matrixTrial][3][4] += row.CURRENT_FIX_DURATION
                        # Increase the sum of the saccade amplitudes in the sentence
                        transM[matrixTrial][3][5] += row.PREVIOUS_SAC_AMPLITUDE

        # If the trial has changed
        else:

            # Update that the trial has changed
            previousTrial = row.trialid

            # Also update the previousIA because the trial has changed
            previousIA = row.IA_group_label

        # If we are at the last row of the dataframe save the results for the last patient
        if row.Index == len(variables) - 1:

            # Save the collected data into the features dataframe --------------------
            chainedList1 = list(chain(*transM))
            chainedListFinal = list(chain(*chainedList1))

            # Insert to the start of the list values to do with the patient
            # Amount of fixations outside the sentences
            #chainedListFinal.insert(0,currentPatientOutFixSum)
            # ZPAF_125_percentile_Group value
            chainedListFinal.insert(0,currentPatientDyslexia)

            features.loc[currentPatient] = chainedListFinal
            # ------------------------------------------------------------------------

    #### If we want the trials on their own rows #############
    elif makeSumMatrix == 3:

        # If the ID of the test subject changes save data to dataframe and prepare for new patient
        if row.id > currentPatient and row.Index != len(variables) - 1:

            # Save the collected data into the features dataframe --------------------
            #chainedListMatrices = list(chain(*transM))
            #print(chainedListMatrices)
            #chainedListFinal = list(chain(*chainedList1))
            i = 2
            for trialMatrix in transM:
                #print(trialMatrix)
                chainedList = list(chain(*trialMatrix))
                #print(chainedList)
                # Insert to the start of the list values to do with the patient
                # Amount of fixations outside the sentences
                #chainedList.insert(0,currentPatientOutFixSum)
                # ZPAF_125_percentile_Group value
                chainedList.insert(0,currentPatientDyslexia)

                currentPatientTrial = "{}T{}".format(currentPatient, i)
                i += 1
                features.loc[currentPatientTrial] = chainedList

            # ------------------------------------------------------------------------
            # RESET VALUES

            currentPatientOutFixSum = 0 # Zero for the next patient
            # Empty the transM matrix for the next patients data
            transM.fill(0)

            # Update current patient and their dyslexia value
            currentPatient = row.id
            currentPatientDyslexia = row[7] #ZPAF_percentile_Group

            # Also correct the previous interest area because the patient changed!
            previousIA = row.IA_group_label

        # If the trial has not changed. This is used to avoid recording wrong information about transitions
        # between the trials.
        if row.trialid == previousTrial:

            # If the current sentence is different than the previous
            if row.IA_group_label != previousIA:

                # The variable for placing the trial details in the 3d matrix
                matrixTrial = row.trialid - 2

                if row.IA_group_label == "first":

                    if previousIA == "task":
                        transM[matrixTrial][1][0] += 1
                    elif previousIA == "distr":
                        transM[matrixTrial][2][0] += 1
                    elif previousIA == "last":
                        transM[matrixTrial][3][0] += 1

                if row.IA_group_label == "task":

                    if previousIA == "first":
                        transM[matrixTrial][0][1] += 1
                    elif previousIA == "distr":
                        transM[matrixTrial][2][1] += 1
                    elif previousIA == "last":
                        transM[matrixTrial][3][1] += 1

                if row.IA_group_label == "distr":

                    if previousIA == "first":
                        transM[matrixTrial][0][2] += 1
                    elif previousIA == "task":
                        transM[matrixTrial][1][2] += 1
                    elif previousIA == "last":
                        transM[matrixTrial][3][2] += 1

                if row.IA_group_label == "last":

                    if previousIA == "first":
                        transM[matrixTrial][0][3] += 1
                    elif previousIA == "task":
                        transM[matrixTrial][1][3] += 1
                    elif previousIA == "distr":
                        transM[matrixTrial][2][3] += 1

                # If the label is empty meaning the fixation went out of the sentences
                if pd.isnull(row.IA_group_label):

                    currentPatientOutFixSum += 1

                previousIA = row.IA_group_label

            # If the current sentence is the same as before
            elif row.IA_group_label == previousIA:

                # The variable for placing the trial details in the 3d matrix
                matrixTrial = row.trialid - 2

                if row.IA_group_label == "first":
                    transM[matrixTrial][0][0] += 1
                    # Increase the sum of the fixation durations in the sentence
                    transM[matrixTrial][0][4] += row.CURRENT_FIX_DURATION
                    # Increase the sum of the saccade amplitudes in the sentence
                    transM[matrixTrial][0][5] += row.PREVIOUS_SAC_AMPLITUDE

                elif row.IA_group_label == "task":
                    transM[matrixTrial][1][1] += 1

                    # Increase the sum of the fixation durations in the sentence
                    transM[matrixTrial][1][4] += row.CURRENT_FIX_DURATION
                    # Increase the sum of the saccade amplitudes in the sentence
                    transM[matrixTrial][1][5] += row.PREVIOUS_SAC_AMPLITUDE

                elif row.IA_group_label == "distr":
                    transM[matrixTrial][2][2] += 1

                    # Increase the sum of the fixation durations in the sentence
                    transM[matrixTrial][2][4] += row.CURRENT_FIX_DURATION
                    # Increase the sum of the saccade amplitudes in the sentence
                    transM[matrixTrial][2][5] += row.PREVIOUS_SAC_AMPLITUDE

                elif row.IA_group_label == "last":
                    transM[matrixTrial][3][3] += 1

                    # Increase the sum of the fixation durations in the sentence
                    transM[matrixTrial][3][4] += row.CURRENT_FIX_DURATION
                    # Increase the sum of the saccade amplitudes in the sentence
                    transM[matrixTrial][3][5] += row.PREVIOUS_SAC_AMPLITUDE

        else:

            # Update that the trial has changed
            previousTrial = row.trialid

            # Also update the previousIA because the trial has changed
            previousIA = row.IA_group_label

        # If we are at the last row of the dataframe save the results for the last patient
        if row.Index == len(variables) - 1:

            # Save the collected data into the features dataframe --------------------
            i = 2
            for trialMatrix in transM:

                chainedList = list(chain(*trialMatrix))
                #print(chainedList)
                # Insert to the start of the list values to do with the patient
                # Amount of fixations outside the sentences
                #chainedList.insert(0,currentPatientOutFixSum)
                # ZPAF_10_percentile_Group value
                chainedList.insert(0,currentPatientDyslexia)

                currentPatientTrial = "{}T{}".format(currentPatient, i)
                i += 1
                features.loc[currentPatientTrial] = chainedList
            # ------------------------------------------------------------------------


###################################################################################

# Test print to see that everything is OK
print("\nBefore normalisation\n", features.head())

# Remove test subjects with trials without data. 121, 362 and 563 left out, as these are unsure cases
#dropList = [105, 142, 175, 186, 207, 383, 396, 472, 508, 525, 531, 543, 545, 547, 549, 552, 560]
dropList = [396]
features = features.drop(dropList)
print("\nAfter removing test subjects\n", features.head())

# Normalisation or feature scaling.
# NOTE: It would also be possible to use sklearns MinMaxScaler
# Calling replace(np.nan, 0) fixes the problem of having a column full of the same numbers
# causing it to be divided by zero in the normalisation resulting in NaN cells. These are
# replaced by zero.
if normalisation:
    scaler = MinMaxScaler(feature_range=(-1,1))
    features = ((features - features.min()) / (features.max() - features.min())).replace(np.nan, 0)
    #features = pd.DataFrame(scaler.fit_transform(features.replace(np.nan, 0)), columns=features.columns).replace(np.nan, 0)
    # After normalisation
    print("\nAfter normalisation\n", features.head())

# Transfer the trial data to rows #####################################################################
if trialRows:
    chosenRow = features.iloc[0,0:25]
    print(chosenRow)
    chosenRow = features.loc[101,"dyslexia":"T2L-L"]
    print(chosenRow)

    # More columns will be added after these
    featuresColumns = ["dyslexia"]

    for s in sentences:

            for e in sentences:
                featuresColumns.append("{}-{}".format(s,e))

            featuresColumns.append("{}FixSum".format(s))
            featuresColumns.append("{}SaccAmplSum".format(s))

    test = pd.DataFrame(columns=featuresColumns)


    for row in features.itertuples():

        for i in range(2,12):

            currentPatientTrial = "{}T{}".format(row.Index, i)
            # Insert the trial data
            trialData = features.loc[row.Index,"T{}F-F".format(i):"T{}LSaccAmplSum".format(i)].tolist()
            trialData.insert(0,row.dyslexia)
            #trialDataFinal = pd.concat([pd.Series({"dyslexia": row.dyslexia}), trialData])
            #print(trialData)
            test.loc[currentPatientTrial] = trialData

    print(test)
#########################################################################################################

# Create scatterplot matrix of the data
#plotting.scatter_matrix(features, c=features.dyslexia)
#sns.pairplot(features, hue="dyslexia", palette=["red", "green"], markers=["x", "o"])

#plt.show()
featuresName = "error"

if makeSumMatrix == 2:
    featuresName = "featuresMatrixAvg"

elif makeSumMatrix == 1:

    if histograms:
        featuresName = "featuresMatrixHistTDrop396"
    else:

        if trialRows:
            featuresName = "featuresMatrixTrialRows"
            features = test
        else:
            featuresName = "featuresMatrix"

elif makeSumMatrix == 3: # This is not currently used
    print("Writing the created feature list into featuresMatrixTrialRows.csv...")
    #features.to_csv("featuresMatrixTrialRows.csv", sep=";", index=True, index_label="id", decimal=',')

# Write the created feature table to a CSV file; include the index
print("Writing the created feature list into {}{}{}.csv...".format(featuresName, zpafBorder, normValue))
features.to_csv("{}{}{}.csv".format(featuresName, zpafBorder, normValue), sep=";", index=True, index_label="id", decimal=',')
