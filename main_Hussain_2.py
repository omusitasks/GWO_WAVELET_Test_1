# Importing modules needed
import pandas as pd
import numpy as np  # Might need this for Albdul's Algorithm
from datetime import datetime # for setting date time data in pandas
import matplotlib.pyplot as plt # Plotting functions
import os
import time


#New imports
import main_GWO
from main_GWO import GWO

start = time.time()

def main_Hussain_2():
    #=======================================================================================================================
    home_dir = "C:/Users/o/OneDrive/Pictures/Desktop/GWO_WAVELET_Test_1/"     # Project home dir
    human_valid = "Human_Validation_TEST1.csv"        # human validation file containing expert assessment
    df1=pd.read_csv(home_dir+human_valid)       # read human validation .csv
    name=df1['Name']        # read first column with names of original freq files
    is_eventH=df1['Is_event']       # read last column that consists of event assessment i.e. True/False

    # Paths for original frequency files
    files_path = "C:/Users/o/OneDrive/Pictures/Desktop/GWO_WAVELET_Test_1/Files_TEST1/"
    #-----------------------------------------------------------------------------------------------------------------------

    is_event2 = []
    file_name = []
    detect = []
    UF = []
    OF = []

    #===================Loop over Code for slew calcs, Event Find Algo======================================================
    for i in range(len(name)):
        f_name=name[i]      # fetch name of frequency file
        file_name.append(f_name) # File name to save for writing out to csv to store Event=T/F & Under/Over frequency
        # Creating the dataframe for the PMU csv file and Parsing data
        df = pd.read_csv(files_path + f_name, delimiter=',',low_memory=False , encoding='latin')
        # Timestamp = df['Timestamp']
        Freq = df['STATION_1:Freq']     # original frequency measurements
        print(file_name)

        # ========================================= Algorithm 5 Parameters =================================================
        slew_window = 158       # window for calculating slew
        point_separation = 3        # gap between two slew points to be compared
        fore_point = point_separation - 1
        slew_diff_thresh = 0.00000385         # slew diff between two slew points separated by "point_separation"
        series_over = 6  # the number of times in a row that slew_diff_thresh is exceeded
        event_thresh = 0.00002161       # deviation of slew at the point when series over is reached
        # ==================================================================================================================

        # Least sum of Squares regression array initiation
        event = 0
        over = 0
        check = 0
        sum_ts = []
        sum_freq = []
        sum_tsfreq = []
        sum_ts2 = []
        slew = []
        intercept = []
        line = []
        ts_ph = np.linspace(0, len(Freq), len(Freq))
        Dif_slew = []
        uf = 0
        of = 0

        for j in range(slew_window, len(Freq)):
            sum_ts.append(np.sum(ts_ph[j - slew_window:j]))
            sum_freq.append(np.sum(Freq[j - slew_window:j]))
            sum_tsfreq.append(np.sum(ts_ph[j - slew_window:j] * Freq[j - slew_window:j]))
            sum_ts2.append(np.sum(ts_ph[j - slew_window:j] * ts_ph[j - slew_window:j]))
            slew.append((slew_window * sum_tsfreq[j-slew_window] - sum_ts[j-slew_window] * sum_freq[j-slew_window]) / (slew_window * sum_ts2[j-slew_window] - sum_ts[j-slew_window] * sum_ts[j-slew_window]))
            intercept.append((sum_freq[j-slew_window] - slew[j-slew_window] * sum_ts[j-slew_window]) / slew_window)
            line.append(slew[j-slew_window] * ts_ph[j-slew_window] + intercept[j-slew_window])

            # Can't compare two slew points unless there are slew values = point_separation

            if j >= slew_window and j < (slew_window + point_separation):
                continue
            # else:
            #     if j + point_separation > (len(Freq) - slew_window):
            #         break
            #     else:

            # Calculate difference of slew between two slew points
            Dif_slew.append(abs(slew[j-slew_window] - slew[(j-slew_window) - fore_point]))


            # can only detect event if there are more than one slew difference values to compare
            if len(Dif_slew) == 1:
                continue

            # Check if slew_diff_thresh is exceeded
            if Dif_slew[(j-slew_window) - point_separation] > slew_diff_thresh and Dif_slew[((j-slew_window) - point_separation) - 1] > slew_diff_thresh:
                over = over + 1
                check = 0
            else:
                check += 1

            # Check is used to reset "over" counter if slew_diff_thresh is not exceeded for a number of consecutive points.
            if check == 2:
                over = 0

            # Detect event based on series_over and event_thresh
            if over >= series_over and abs(slew[j - slew_window] - 0.00) >= event_thresh:
                    event = event + 1
                    detect.append(j)      # save the sample at which event is detected
                    #print(event, detect)

            # if slew[j-1] > slew[j]:
            #     uf=1
            #     of=0
            # elif slew[j-1] < slew[j]:
            #     of=1
            #     uf=0
            # else:
            #     uf=0
            #     of=0

            if event==1:
                is_event2.append(True)  # This will save the EVENT true classification
                print(is_event2)


        print("**********************************************************************")
                # UF.append(uf)
                # OF.append(of)
                #break

        if len(is_event2) != i+1:
            is_event2.append(False)
            # UF.append(uf)
            # OF.append(of)

    # ====================== Evaluation code =======================================
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(is_event2)):
        if is_event2[i] == is_eventH[i] and is_event2[i] == True:
            TP=TP+1
        elif is_event2[i] == is_eventH[i] and is_event2[i] == False:
            TN=TN+1
        elif is_event2[i] != is_eventH[i] and is_event2[i] == True:
            FP=FP+1
        elif is_event2[i] != is_eventH[i] and is_event2[i] == False:
            FN=FN+1


    totalexample = TP + FP + FN + TN
    print("TP = {}\nFP = {}\nFN = {}\nTN = {}\nTotal examples = {}".format(TP, FP, FN, TN, totalexample))
    Accuracy = round(((TP + TN) / totalexample) * 100)
    if TP == 0 and FN == 0:
        FN = totalexample
    Sensitivity = (TP / (TP + FN)) * 100
    if TP == 0 and FP == 0:
        FP = totalexample
    Precision = round((TP / (TP + FP)) * 100)
    if TN == 0 and FP == 0:
        FP = totalexample
    Specificity = round((TN / (TN + FP)) * 100)
    FDR = round((FP / (FP + TP)) * 100)
    # calculating the F scores
    print('Accuracy:',Accuracy)
    print('Sensitivity:',Sensitivity)
    print('Precision:',Precision)
    print('Specificity:',Specificity)
    print('FDR:',FDR)


    end = time.time()
    print('Time elapsed:', end-start)



start = time.time()
SearchAgents_no = 10
Max_iteration = 30
main_GWO.data_handle()  # read csv files and shrink frequency files by removing normal freq data
lb, ub, dim = main_GWO.Get_Function_details()
# Best_score, Best_pos, GWO_cg_curve = GWO(SearchAgents_no, Max_iteration, lb, ub, dim, fobj)
Best_score, Best_pos = GWO(main_Hussain_2, SearchAgents_no, Max_iteration, lb, ub, dim)
print("Best Fitness: %.2f" % Best_score)
print("Window Size: ", int(Best_pos[0]))#1
print("Point Separation: " + "%.8f" % Best_pos[1]) #2
print("Slew Difference Threshold: " + "%.8f" % Best_pos[2])#3
print("Series Over: ", int(Best_pos[3]))#4
print("Event Threshold: " + "%.8f" % Best_pos[4])#5
end = time.time()
print('Time elapsed:', end - start, "sec")

