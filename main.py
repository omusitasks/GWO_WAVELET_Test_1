import numpy as np
import random
import pandas as pd
import time

#New imports
import main_Hussain_1
from main_Hussain_1 import main_Hussain_1



# This function reads the original frequency files and calculates slew rate.
def data_handle():
    home_dir = "C:/Users/o/OneDrive/Pictures/Desktop/GWO_WAVELET_Test_1/"  # N:\Desktop\GWO_WAVELET_Test 1\Files_TEST1
    human_valid = "Human_Validation_TEST1.csv"  # human validation file namez Desktop\GWO_WAVELET_Test 1
    df1 = pd.read_csv(home_dir + human_valid)  # read human validation .csv
    global name, is_eventH
    name = df1['Name']  # read first column with names of original freq files
    is_eventH = df1['Is_event']  # read last column that consists of event assessment i.e. True/False
    files_path = "C:/Users/o/OneDrive/Pictures/Desktop/GWO_WAVELET_Test_1/Files_TEST1/"  # original frequency files directory #""



def GWO(objf,lb,ub,dim,SearchAgents_no,Max_iter):
    
    # initialize alpha, beta, and delta_pos
    Alpha_pos=np.zeros(dim)
    Alpha_score=float("inf")
    
    Beta_pos=np.zeros(dim)
    Beta_score=float("inf")
    
    Delta_pos=np.zeros(dim)
    Delta_score=float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    #Initialize the positions of search agents
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.uniform(0,1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
    
    Convergence_curve=np.zeros(Max_iter)

     # Loop counter
    # print("GWO is optimizing  \""+objf.__name__+"\"")    

    # Main loop
    for l in range(0,Max_iter):
        for i in range(0,SearchAgents_no):
            
            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i,j]=np.clip(Positions[i,j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness=objf(Positions[i,:])
            
            # Update Alpha, Beta, and Delta
            if fitness<Alpha_score :
                Alpha_score=fitness; # Update alpha
                Alpha_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness<Beta_score ):
                Beta_score=fitness  # Update beta
                Beta_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness>Beta_score and fitness<Delta_score): 
                Delta_score=fitness # Update delta
                Delta_pos=Positions[i,:].copy()
        
        a=2-l*((2)/Max_iter); # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
        for i in range(0,SearchAgents_no):
            for j in range (0,dim):     
                           
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                
                A1=2*a*r1-a; # Equation (3.3)
                C1=2*r2; # Equation (3.4)
                
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]); # Equation (3.5)-part 1
                X1=Alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1
                           
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a; # Equation (3.3)
                C2=2*r2; # Equation (3.4)
                
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); # Equation (3.5)-part 2
                X2=Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2       
                
                r1=random.random()
                r2=random.random() 
                
                A3=2*a*r1-a; # Equation (3.3)
                C3=2*r2; # Equation (3.4)
                
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
                X3=Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3             
                
                Positions[i,j]=(X1+X2+X3)/3  # Equation (3.7)        
        Convergence_curve[l]=Alpha_score

        #if (l%1==0):
               #print(['At iteration '+ str(l)+ ' the best fitness is '+ str(Alpha_score)]);
    
    print(Positions.shape)
    print("Alpha position=",Alpha_pos)
    print("Beta position=",Beta_pos)
    print("Delta position=",Delta_pos)
    return Alpha_pos,Beta_pos

# Setting GWO parameters
iters=10
wolves=5
dimension=13
search_domain=[0,1]
lb=-1.28
ub=1.28
start = time.time()
alpha, beta = GWO(main_Hussain_1, lb,ub,dimension,wolves,iters)
print("Best Fitness: %.2f" % alpha)
print("Window Size parameter: ", int(beta[0]))#1
print("Fre Measurement Difference parameter: " + "%.8f" % beta[2])#2
print("Over series SD parameter: ", int(beta[3]))#3
print("Upper Threshold SD parameter: " + "%.8f" % beta[4])#4

end = time.time()
print('Time elapsed:', end - start, "sec")