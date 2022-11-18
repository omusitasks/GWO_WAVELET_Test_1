import datetime as datetime
import pandas as pd
import numpy as np
from datetime import datetime  # for setting date time data in pandas
import matplotlib.pyplot as plt  # Plotting functions
import math
import os
import time
import random



# This function reads the original frequency files and calculates slew rate.
def data_handle():
    home_dir = "C:/Users/o/OneDrive/Pictures/Desktop/GWO_WAVELET_Test_1/"  # N:\Desktop\GWO_WAVELET_Test 1\Files_TEST1
    human_valid = "Human_Validation_TEST1.csv"  # human validation file namez Desktop\GWO_WAVELET_Test 1
    df1 = pd.read_csv(home_dir + human_valid)  # read human validation .csv
    global name, is_eventH
    name = df1['Name']  # read first column with names of original freq files
    is_eventH = df1['Is_event']  # read last column that consists of event assessment i.e. True/False
    files_path = "C:/Users/o/OneDrive/Pictures/Desktop/GWO_WAVELET_Test_1/Files_TEST1/"  # original frequency files directory #""

    file_name = []
    freq_original = []
    slew = []

    for i in range(len(name)):
        f_name = name[i]  # fetch frequency file name
        print(f_name)
        file_name.append(f_name)  # FIle name to save for writing out to csv to store Event=T/F & Under/Over frequency
        # # Creating the dataframe for the PMU csv file and Parsing data
        df = pd.read_csv(files_path + f_name, delimiter=',', low_memory=False,encoding='latin1') # encoding='latin1' was add it by hussain to solve encoding error
        # Timestamp = df['Timestamp']
        freq_original.append(df['STATION_1:Freq'])  # original frequency measurements
        ts_ph = np.linspace(0, len(freq_original[i]), len(freq_original[i]))

        # This slew window is chosen just to calculate slew to help find disturbance in each file and remove normal freq data
        window = 350
        sum_ts = []
        sum_freq = []
        sum_tsfreq = []
        sum_ts2 = []
        slew_temp = []
        intercept = []
        line = []

        # Slew calculation
        for j in range(len(freq_original[i])):
            if j + window > len(freq_original[i]):
                break
            else:
                sum_ts.append(np.sum(ts_ph[j:j + window]))
                sum_freq.append(np.sum(freq_original[i][j:j + window]))
                sum_tsfreq.append(np.sum(ts_ph[j:j + window] * freq_original[i][j:j + window]))
                sum_ts2.append(np.sum(ts_ph[j:j + window] * ts_ph[j:j + window]))
                slew_temp.append((window * sum_tsfreq[j] - sum_ts[j] * sum_freq[j]) / (
                        window * sum_ts2[j] - sum_ts[j] * sum_ts[j]))
                intercept.append((sum_freq[j] - slew_temp[j] * sum_ts[j]) / window)
                line.append(slew_temp[j] * ts_ph[j] + intercept[j])
        slew.append(slew_temp)

    data_shrink(freq_original, slew)  # Send original freq and corresponding slew to remove normal freq data


# This function shrinks the freq files by removing normal freq data to reduce processing time
def data_shrink(freq_original, slew):
    global Freq
    Freq = []
    for i in range(len(slew)):
        freq_temp = []
        maximum = max(slew[i])  # find maximum slew value in each file
        minimum = min(slew[i])  # find minimum slew value in each file
        f = freq_original[i]
        max_index = list(slew[i]).index(maximum)  # index of maximum value
        min_index = list(slew[i]).index(minimum)  # index of minimum value

        x1_max = max_index - 3000  # take 3000 samples before the maximum index
        x2_max = max_index + 500  # take 500 samples after the maximum index
        if x1_max < 0:  # Set x1_max = 0 if index of max value is in the beginning of the file
            x1_max = 0
        if x2_max > len(f) - 1:  # set x2_max = max index if index of max value is at the end
            x2_max = len(f) - 1

        x1_min = min_index - 3000  # same as above
        x2_min = min_index + 500
        if x1_min < 0:
            x1_min = 0
        if x2_min > len(f) - 1:
            x2_min = len(f) - 1

        if abs(maximum - 0.0) > abs(minimum - 0.0):  # if max slew value is greater than min slew value in a file
            freq_temp = f.iloc[x1_max:x2_max]  # keep only the frequency data around the maximum index

        elif abs(maximum - 0.0) <= abs(minimum - 0.0):  # if min slew value is greater than max slew value in a file
            freq_temp = f.iloc[x1_min:x2_min]  # keep only the data around minimum index

        Freq.append(freq_temp)  # Freq holds the shrunk freq files after removing normal freq data


# Memoizaiton
def detect_event(parameters, ts_ph, Freq, file_name, memo={}):
    try:
        return memo[(parameters[0], parameters[1], parameters[2], parameters[3], file_name)]
    except KeyError:
        event = 0
        over = 0
        check = 0
        window = int(parameters[0])
        point_separation = int(parameters[1])
        event_param = parameters[2]
        over_series_SD = int(parameters[3])
        floor = parameters[4]
        fore_point = point_separation - 1

        sum_ts = []
        sum_freq = []
        sum_tsfreq = []
        sum_ts2 = []
        slew = []
        intercept = []
        line = []
        Dif_slew = []
        # uf = 0
        # of = 0

        # Calculate slew and look for events
        for j in range(len(Freq)):
            if j + window > len(Freq):
                break
            else:
                sum_ts.append(np.sum(ts_ph[j:j + window]))
                sum_freq.append(np.sum(Freq[j:j + window]))
                sum_tsfreq.append(np.sum(ts_ph[j:j + window] * Freq[j:j + window]))
                sum_ts2.append(np.sum(ts_ph[j:j + window] * ts_ph[j:j + window]))
                slew.append((window * sum_tsfreq[j] - sum_ts[j] * sum_freq[j]) / (
                        window * sum_ts2[j] - sum_ts[j] * sum_ts[j]))
                intercept.append((sum_freq[j] - slew[j] * sum_ts[j]) / window)
                line.append(slew[j] * ts_ph[j] + intercept[j])

                if j >= 0 and j < point_separation:
                    continue  # next sample

                Dif_slew.append(abs(slew[j] - slew[j - fore_point]))  # Slew difference

                if len(Dif_slew) == 1:
                    continue  # move to next iteration and calculate another slew diff

                if Dif_slew[j - point_separation] > event_param and Dif_slew[(j - point_separation) - 1] > event_param:
                    over = over + 1  # over_series_SD increment
                    check = 0
                else:
                    check += 1  # increment if slew diff is small (freq becoming normal)

                if check == 2:  # Check if sle diff is normal for two consecutive values
                    over = 0  # reset over_series_SD. It is needed to reset if freq becomes normal

                # Event detection condition
                if over >= over_series_SD and abs(slew[j] - 0.00) >= floor:
                    event = event + 1  # event detected
                # if slew[j - 1] > slew[j]:
                #     uf = 1
                #     of = 0
                # elif slew[j - 1] < slew[j]:
                #     of = 1
                #     uf = 0
                # else:
                #     uf = 0
                #     of = 0

                if event == 1:
                    break  # break loop if event detected

    memo[(parameters[0], parameters[1], parameters[2], parameters[3], file_name)] = event
    print(parameters[0], parameters[1], parameters[2], parameters[3]) #Hussain
    return event


# problem dimension and boundary of solution space
def Get_Function_details():
    lb = [87, 7, 0.000013, 9, 0.000018]  # for 30 samples/sec, for fast convergence
    ub = [105, 3, 0.000021, 27, 0.00015]

    dim = 5
    return lb, ub, dim


def object_function(position):
    is_event = []
    # UF = []
    # OF = []

    for i in range(len(name)):

        ts_ph = np.linspace(0, len(Freq[i]), len(Freq[i]))
        event = detect_event(position, ts_ph, Freq[i], name[i])

        if event == 1:
            is_event.append(True)  # This will save the EVENT true classification
            # UF.append(uf)
            # OF.append(of)
        else:
            is_event.append(False)  # This will save the EVENT false classification
            # UF.append(uf)
            # OF.append(of)

    # ====================== Evaluation code =======================================
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(is_event)):
        if is_event[i] == is_eventH[i] and is_event[i] == True:
            TP = TP + 1
        elif is_event[i] == is_eventH[i] and is_event[i] == False:
            TN = TN + 1
        elif is_event[i] != is_eventH[i] and is_event[i] == True:
            FP = FP + 1
        elif is_event[i] != is_eventH[i] and is_event[i] == False:
            FN = FN + 1

    totalexample = TP + FP + FN + TN
    Accuracy = ((TP + TN) / totalexample) * 100
    if TP == 0 and FN == 0:
        FN = totalexample
    Sensitivity = (TP / (TP + FN)) * 100
    if TP == 0 and FP == 0:
        FP = totalexample
    Precision = round((TP / (TP + FP)) * 100)
    if TN == 0 and FP == 0:
        FP = totalexample
    Specificity = round((TN / (TN + FP)) * 100)

########################################  Just For Explaination Not A Part Of The GWO  #################################
    fitness = (Accuracy) + (Sensitivity) + (Precision) + (Specificity) #Hussain
    # Accuracy + Sensitivity + Precision + Specificity
    print(Accuracy, Sensitivity, Specificity, Precision) #Hussain
    print(TP, TN, FN, FP) #Hussain
    print(fitness) #Hussain
########################################################################################################################

    return fitness


# GWO Algorithm
def GWO(objf,SearchAgents_no, Max_iter, lb, ub, dim):
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("-inf")  # Set +inf for minimization problem

    Beta_pos = np.zeros(dim)
    Beta_score = float("-inf")

    Delta_pos = np.zeros(dim)
    Delta_score = float("-inf")

    # Initialize the positions of search agents
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]

    Convergence_curve = np.zeros(Max_iter)

    for l in range(Max_iter):
        for i in range(SearchAgents_no):
            for j in range(dim):
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])  # Enforce search space limits

            fitness = object_function(Positions[i, :])  # Calculate fitness

            # Update Alpha, Beta, and Delta
            if fitness > Alpha_score:
                Alpha_score = fitness  # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if (fitness < Alpha_score and fitness > Beta_score):
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if (fitness < Beta_score and fitness > Delta_score):
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        # if Alpha_score >= 380:
        #     break

        # Loop counter
        print("GWO is optimizing  \""+object_function.__name__+"\"") 

        a = 2 - l * (2 / Max_iter)  # update GWO parameter a

        for i in range(SearchAgents_no):
            for j in range(dim):
                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A_alpha = 2 * a * r1 - a  # Eq 3.3 in GWO paper
                C_alpha = 2 * r2  # Eq 3.4 in GWO paper

                D_alpha = abs(C_alpha * Alpha_pos[j] - Positions[i, j])  # Eq 3.5 in GWO paper
                X1 = Alpha_pos[j] - A_alpha * D_alpha  # Eq 3.6 in GWO paper

                A_beta = 2 * a * r1 - a  # Eq 3.3 in GWO paper
                C_beta = 2 * r2  # Eq 3.4 in GWO paper

                D_beta = abs(C_beta * Beta_pos[j] - Positions[i, j])  # Eq 3.5 in GWO paper
                X2 = Beta_pos[j] - A_beta * D_beta  # Eq 3.6 in GWO paper

                A_delta = 2 * a * r1 - a  # Eq 3.3 in GWO paper
                C_delta = 2 * r2  # Eq 3.4 in GWO paper

                D_delta = abs(C_delta * Delta_pos[j] - Positions[i, j])  # Eq 3.5-part 3
                X3 = Delta_pos[j] - A_delta * D_delta  # Eq 3.5-part 3

                # Update positions of all wolves according to alpha, beta, and delta
                Positions[i, j] = (X1 + X2 + X3) / 3  # Eq 3.7 in GWO paper

        Convergence_curve[l] = Alpha_score  # Convergence curve for best score

        if (l % 1 == 0):
            print(['At iteration ' + str(l + 1) + ' the best fitness is ' + str(Alpha_score)])

    # Return best score and corresponding best position
    return Alpha_score, Alpha_pos