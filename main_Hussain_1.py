import pandas as pd
import numpy as np
import math
import statistics
from dateutil import parser  # importing parser
from scipy.signal import savgol_filter as Filter
from scipy.io import wavfile
import skimage
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import matplotlip.pyplot            #matplotlip.pyplot as plt
# import matplotlib.ticker as ticker
import os
import datetime as datetime
from datetime import datetime  # for setting date time data in pandas
import time
import codecs

#New imports
import main_GWO1
from main_GWO1 import GWO


def main_Hussain_1():
    ################################## from umar detection code with 5 para #################################

    home_dir = "C:/Users/o/OneDrive/Pictures/Desktop/GWO_WAVELET_Test_1/"     # Project home dir
    human_valid = "Human_Validation_TEST1.csv"        # human validation file containing expert assessment
    df1=pd.read_csv(home_dir+human_valid)       # read human validation .csv
    name=df1['Name']        # read first column with names of original freq files
    is_eventH=df1['Is_event']       # read last column that consists of event assessment i.e. True/False

    # # Paths for original frequency files
    files_path = "C:/Users/o/OneDrive/Pictures/Desktop/GWO_WAVELET_Test_1/Files_TEST1/"

    is_event2 = []
    file_name = []


    ###############################################   Algorithm Four Parameters    #########################################

    # Freq Measurements Difference Parameter
    diff = 5.54

    # Window Size Parameter
    window = 250

    # upper_threshold_SD Parameter
    upper_threshold_SD = 0.0015

    # over_series_SD Parameter
    over_series_SD = 12


    ###############################################      ROCOF Setting Points      #########################################

    # ROCOF Setting Points
    ROCOF_lowerThreshold = -0.013
    ROCOF_upperThreshold = 0.023
    ROCOF_Event_series_over = 60



    ########################   First Loop For Reading CSV files And Denoising Frequency Signal   ###########################

    for i in range(len(name)):
        f_name=name[i]      # fetch name of frequency file
        file_name.append(f_name) # File name to save for writing out to csv to store Event=T/F & Under/Over frequency
        # Creating the dataframe for the PMU csv file and Parsing data
        df = pd.read_csv(files_path + f_name, delimiter=',',low_memory=False , encoding='latin')
        # df = pd.read_csv(path1, low_memory=False)
        # Timestamp = df['Timestamp']
        Freq = df['STATION_1:Freq']     # original frequency measurements
        print("test1", file_name)



    ##############################    Denoising Frequency Signal Using Wavelet Transform    ################################

        # Wavelet denoising
        # denoise_freq = denoise_wavelet(noise_freq, method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym2', rescale_sigma='True')
        # y = denoise_wavelet(x, wavelet='db1', mode='soft', wavelet_levels=n, method='BayesShrink', rescale_sigma='True')
        # denoise_freq = denoise_wavelet(noise_freq, method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='db1', rescale_sigma='True')

        noise_freq = Freq
        denoise_freq = denoise_wavelet(noise_freq, method='BayesShrink', mode='soft', wavelet_levels=5,
                                    wavelet='db4', rescale_sigma='True')

        # Read csv file
        def readData(ReadMe):
            df = denoise_freq
            return df

        # time difference
        def getTimeDifference(t1, t2):
            firstTime = parser.parse(t1)
            seconTime = parser.parse(t2)
            return (seconTime - firstTime).total_seconds()

        # frequency difference
        def getFrequencyDiff(f1, f2):
            return f2 - f1

        # Calculate Rate Of Change Of Frequency (ROCOF)
        # def getDifference(dff, diff, lowerThreshold, upperThreshold):
        ROCOF_List = []
        A = ROCOF_List
        ROCOF_noise_List = []
        # A = result_noise_List=[]
        ROCOF_lowerCount = 0
        ROCOF_upperCount = 0

        

    ###########################################   Second Loop For ROCOF Calculation   ######################################

        for i in range(1, len(df['Timestamp'])):
            if (i + diff < len(df['Timestamp'])):
                time = getTimeDifference(df['Timestamp'][i - 1], df['Timestamp'][i + diff])
                freq = getFrequencyDiff(denoise_freq[i - 1], denoise_freq[i + diff])
                freq_noise = getFrequencyDiff(df['STATION_1:Freq'][i - 1], df['STATION_1:Freq'][i + diff])

                if (time != 0):
                    ROCOF = freq / time                             # Denoise (ROCOF)
                    ROCOF_noise = freq_noise / time                 # Noise (ROCOF)

                    if ROCOF < ROCOF_lowerThreshold:
                        ROCOF_lowerCount += 1

                    if ROCOF > ROCOF_upperThreshold:
                        ROCOF_upperCount += 1

                    if ROCOF_lowerCount > ROCOF_Event_series_over:
                        ROCOF_lowerCount = 0

                    if ROCOF_upperCount > ROCOF_Event_series_over:
                        ROCOF_upperCount = 0

                    ROCOF_List.append(freq / time)                   # Denoise (ROCOF) List
                    ROCOF_noise_List.append(freq_noise / time)       # Noise (ROCOF) List

        list_sum_means = []
        mean5 = []
        variance5 = []
        variance_list = []
        j1 = []
        StandardDeviation_equation2 = []
        variance_equation2 = []
        Event_Declaration = 0
        counter_high_SD = 0
        counter_low_SD = 0


    #########################  Third Loop For Standard Deviation Calculation And Declare The Event   #######################

        for j in range(window, len(A)):
            j1.append(j)
            mean4 = np.sum(A[j - window:j]) / len(A[j - window:j])
            variance_equation2.append(
                np.sum(((A[j - window:j]) - (np.sum(A[j - window:j]) / len(A[j - window:j]))) ** 2) / len(
                    A[j - window:j]))
            StandardDeviation_equation2.append(math.sqrt(
                np.sum(((A[j - window:j]) - (np.sum(A[j - window:j]) / len(A[j - window:j]))) ** 2) / len(
                    A[j - window:j])))
            SD = (math.sqrt(
                np.sum(((A[j - window:j]) - (np.sum(A[j - window:j]) / len(A[j - window:j]))) ** 2) / len(
                    A[j - window:j])))

            if SD > upper_threshold_SD:
                counter_high_SD += 1

            if counter_high_SD > over_series_SD:
                print('high SD event', j)
                counter_high_SD = 0
                Event_Declaration += 1
                print(Event_Declaration)

            if Event_Declaration == 1:
                is_event2.append(True)
                break
        else:
            is_event2.append(False)

            # if len(is_event2) != j + 1:
            # is_event2.append(False)

        print("is_event2:", is_event2)
        print("**************************************************************************************************")


    ################################################ Evaluation code #######################################################

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


    ############################################# calculating the F scores #################################################

    print('Accuracy:',Accuracy)
    print('Sensitivity:',Sensitivity)
    print('Precision:',Precision)
    print('Specificity:',Specificity)
    print('FDR:',FDR)

start = time.time()
SearchAgents_no = 5
Max_iteration = 5
main_GWO1.data_handle()  # read csv files and shrink frequency files by removing normal freq data
lb, ub, dim = main_GWO1.Get_Function_details()
# alpha, Best_pos, GWO_cg_curve = GWO(SearchAgents_no, Max_iteration, lb, ub, dim, fobj)
alpha, beta = GWO(main_Hussain_1, SearchAgents_no, Max_iteration, lb, ub, dim)
print("Best Fitness: %.2f" % alpha)
print("Window Size parameter: ", int(beta[0]))#1
print("Fre Measurement Difference parameter: " + "%.8f" % beta[2])#2
print("Over series SD parameter: ", int(beta[3]))#3
print("Upper Threshold SD parameter: " + "%.8f" % beta[4])#4
end = time.time()
print('Time elapsed:', end - start, "sec")

