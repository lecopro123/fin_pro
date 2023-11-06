import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm, rayleigh, expon,gaussian_kde
from scipy import special as sp
from numpy import asarray


MHT=4  ###maximum HARQ repetitions
num_trials=10
MNU=3  ####maximum number of users
snr_db = 16
L = 3  ###rx at BS
#########################
# M = 2  #########BPSK constellation constant
# m = np.arange(0, M)
# constellation = 1 * np.cos(m / M * 2 * np.pi)
# stream_0_1 = np.random.randint(low=0, high=M, size=num_trials)
# ones_and_minus_ones = constellation[stream_0_1]
###################

for i in range(num_trials):
    [i,j] = np.random.randint(low=0, high=MNU, size=2)
    M_co=0 #####this means that all the HARQ retransmissions have been extinguised
    #print(i,j)
    if(i==j):
        #########BPSK constellation constant
        M = 2  
        m = np.arange(0, M)
        constellation = 1 * np.cos(m / M * 2 * np.pi)
        stream_0_1 = np.random.randint(low=0, high=M, size=MHT)    ##########this steps ensure that the max Harq retransmissions are 4.
        ones_and_minus_ones = constellation[stream_0_1]
        ###########################################
        mean_x1=1
        var_x1 = 0.5
        mean_x2=0
        var_x2= 1.5
        mean_x3=0
        var_x3 = 2.5
        noise1 = norm.rvs(loc=mean_x1, scale=var_x1, size=1)
        noise2 = norm.rvs(loc=mean_x1, scale=var_x1, size=1)
        noise3 = norm.rvs(loc=mean_x2, scale=var_x2, size=1)
        noise4 = norm.rvs(loc=mean_x2, scale=var_x2, size=1)
        noise5 = norm.rvs(loc=mean_x3, scale=var_x3, size=1)
        noise6 = norm.rvs(loc=mean_x3, scale=var_x3, size=1)
        h1 = norm.rvs(loc=mean_x1, scale=var_x1,
                    size=1)  ####channel coefficients
        h2 = norm.rvs(loc=mean_x1, scale=var_x1,
                    size=1)  ## channel coefficients
        h3 = norm.rvs(loc=mean_x2, scale=var_x2,
                    size=1)  #### channel coefficients
        h4 = norm.rvs(loc=mean_x2, scale=var_x2,
                    size=1)  ####channel coefficients
        h5 = norm.rvs(loc=mean_x3, scale=var_x3,
                    size=1)  ## channel coefficients
        h6 = norm.rvs(loc=mean_x3, scale=var_x3,
                    size=1)  #### channel coefficients
        #################CSI at receiver
        H1=h1+h2*1j
        H2=h3+h4*1j
        H3=h5+h6*1j
        w1 = H1 / np.sqrt(abs(H1)**2 + abs(H2)**2 + abs(H3)**2)
        w2 = H2 / np.sqrt(abs(H1)**2 + abs(H2)**2 + abs(H3)**2)
        w3 = H3 / np.sqrt(abs(H1)**2 + abs(H2)**2 + abs(H3)**2)
        # w1 = np.conj(h1) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
        # w2 = np.conj(h2) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
        # w3 = np.conj(h3) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
        ##########################
        P = 1
        snr = 10**(snr_db / 10)
        N0 = P / snr
        recv_1 = (H1 * ones_and_minus_ones) + (noise1+1j*noise2) * np.sqrt(N0 / 2)
        recv_2 = (H2 * ones_and_minus_ones) + (noise3+1j*noise4) * np.sqrt(N0 / 2)
        recv_3 = (H3 * ones_and_minus_ones) + (noise5+1j*noise6) * np.sqrt(N0 / 2)
        # recv_1 = (h1 * ones_and_minus_ones) + noise1 * np.sqrt(N0 / 2)
        # recv_2 = (h2 * ones_and_minus_ones) + noise2 * np.sqrt(N0 / 2)
        # recv_3 = (h3 * ones_and_minus_ones) + noise3 * np.sqrt(N0 / 2)
        #########################Combination
        comb_at_recv = np.conj(w1) * recv_1 + np.conj(w2) * recv_2 + np.conj(w3) * recv_3
        print(comb_at_recv)
        print(ones_and_minus_ones)
        dec=[]
        for i in comb_at_recv:
            if (np.real(i) >0):
                dec.append(0)
            else:
                dec.append(1)

        ################################
        for i in range(1, len(stream_0_1)):
            if dec[i] != stream_0_1[i]: 
                M_co = M_co + 1
            # c_mc.append(M_co / i)
            # x_mc.append(i)
        print(M_co)    
#print(count / num_trials,"Van MC")
    
            