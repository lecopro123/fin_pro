#####same packet will be sent in HARQ retx
#####collision of packets is not considerd as a event where HARQ retx is extinguised
#####Grant Access Failure: In M HARQ tx the UE is unable to take access of the BS

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm, rayleigh, expon,randint
from scipy import special as sp
from numpy import asarray


MHT=4  ###maximum HARQ repetitions
num_trials=100000
MNU=3  ####maximum number of users
snr_db = -8
L = 3  ###rx at BS
snr = 10**(snr_db / 10)
th=1/snr
#########################
# M = 2  #########BPSK constellation constant
# m = np.arange(0, M)
# constellation = 1 * np.cos(m / M * 2 * np.pi)
# stream_0_1 = np.random.randint(low=0, high=M, size=num_trials)
# ones_and_minus_ones = constellation[stream_0_1]
###################
gf=0
#c=0
H1=[]
H2=[]
H3=[]
M_cos=[]
for a in range(num_trials):
    M_co=0 #########this is the termination
    while M_co<=MHT:
        #if(M_co==4): print("Hi")
        i = randint.rvs(low=0, high=MNU, size=1)
        j = randint.rvs(low=0, high=MNU, size=1)
        #print(i,j)
        if(i[0]==j[0]):
            #########BPSK constellation constant
            M = 2  
            m = np.arange(0, M)
            constellation = 1 * np.cos(m / M * 2 * np.pi)
            stream_0_1 = np.random.randint(low=0, high=M, size=1)    ##########this steps ensure that the max Harq retransmissions are 4.
            ones_and_minus_ones = constellation[stream_0_1]
            ###########################################
            mean_x1=1
            var_x1 = 0.5
            mean_x2=2
            var_x2= 0.5
            mean_x3=0
            var_x3 = 1.5
            noise1 = norm.rvs(loc=mean_x1, scale=var_x1, size=1)
            noise2 = norm.rvs(loc=mean_x1, scale=var_x1, size=1)
            noise3 = norm.rvs(loc=mean_x2, scale=var_x2, size=1)
            # noise4 = norm.rvs(loc=mean_x2, scale=var_x2, size=1)
            # noise5 = norm.rvs(loc=mean_x3, scale=var_x3, size=1)
            # noise6 = norm.rvs(loc=mean_x3, scale=var_x3, size=1)
            h1 = norm.rvs(loc=mean_x1, scale=var_x1,
                        size=1)  ####channel coefficients
            h2 = norm.rvs(loc=mean_x1, scale=var_x1,
                        size=1)  ## channel coefficients
            h3 = norm.rvs(loc=mean_x2, scale=var_x2,
                        size=1)  #### channel coefficients
            # h4 = norm.rvs(loc=mean_x2, scale=var_x2,
            #             size=1)  ####channel coefficients
            # h5 = norm.rvs(loc=mean_x3, scale=var_x3,
            #             size=1)  ## channel coefficients
            # h6 = norm.rvs(loc=mean_x3, scale=var_x3,
            #             size=1)  #### channel coefficients
            #################CSI at receiver
            
            # w1 = H1 / np.sqrt(abs(H1)**2 + abs(H2)**2 + abs(H3)**2)
            # w2 = H2 / np.sqrt(abs(H1)**2 + abs(H2)**2 + abs(H3)**2)
            # w3 = H3 / np.sqrt(abs(H1)**2 + abs(H2)**2 + abs(H3)**2)
            w1 = np.conj(h1) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
            w2 = np.conj(h2) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
            w3 = np.conj(h3) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
            ##########################
            P = 1
            # snr = 10**(snr_db / 10)
            N0 = P / snr
            # recv_1 = (H1 * ones_and_minus_ones) + (noise1+1j*noise2) * np.sqrt(N0 / 2)
            # recv_2 = (H2 * ones_and_minus_ones) + (noise3+1j*noise4) * np.sqrt(N0 / 2)
            # recv_3 = (H3 * ones_and_minus_ones) + (noise5+1j*noise6) * np.sqrt(N0 / 2)
            recv_1 = (h1 * ones_and_minus_ones) + noise1 * np.sqrt(N0 / 2)
            recv_2 = (h2 * ones_and_minus_ones) + noise2 * np.sqrt(N0 / 2)
            recv_3 = (h3 * ones_and_minus_ones) + noise3 * np.sqrt(N0 / 2)
            #########################Combination
            comb_at_recv = np.conj(w1) * recv_1 + np.conj(w2) * recv_2 + np.conj(w3) * recv_3
            # print(comb_at_recv)
            # print(ones_and_minus_ones)
            dec=0
            for i in comb_at_recv:
                if (np.real(i) >0):
                    dec=0
                else:
                    dec=1
            ################################
            if dec != stream_0_1: ############Counter M++
                M_co = M_co + 1
            else:    ###########conn established
                break
    H1.append(h1[0])
    H2.append(h2[0])
    H3.append(h3[0])
    M_cos.append(M_co)
    #print(M_co)
###MONTE CARLO METHOD           
    if M_co>MHT:  ######Grant access failure: how many times in num_trials M_co>4    
        #print(M_co)    
        gf=gf+1
print(gf/num_trials)
#print(c)
    

a1 = np.log(1 + np.absolute(H1))
a2= np.log(1 + np.absolute(H2))
a3= np.log(1 + np.absolute(H3))
b=np.log(16.9)
exp3=a3/b
exp2=a2/b
exp1=a1/b
new_x1 = H1 * (th /0.7)**exp1  
new_x2 = H2 * (th /0.7)**exp2 
new_x3 = H3 * (th /0.7)**exp3 


weight1=(norm.pdf(new_x1,loc=np.mean(H1),scale=np.var(H1)))/(norm.pdf(new_x1,loc=np.mean(new_x1),scale=np.var(new_x1)))  #weight of 1st dim
weight2=(norm.pdf(new_x2,loc=np.mean(H2),scale=np.var(H2)))/(norm.pdf(new_x2,loc=np.mean(new_x2),scale=np.var(new_x2)))  #weight of 2nd dim
weight3=(norm.pdf(new_x3,loc=np.mean(H3),scale=np.var(H3)))/(norm.pdf(new_x3,loc=np.mean(new_x3),scale=np.var(new_x3)))  #weight of 3rd dim 
# there will be two more weights.the problem params will be abs(new_x1[i])**2+abs(new_x2[i])**2+abs(new_x3[i])**2<th and new_i==new_j

##############SS Importance Sampling
count = 0

# # c_ss_is=[]
# # x_ss_is=[]

for i in range(1, num_trials):
    if abs(new_x1[i])**2+abs(new_x2[i])**2+abs(new_x3[i])**2<th: #and M_cos[i]>MHT:
        count = count + weight1[i]*weight2[i]*weight3[i]
    # c_ss_is.append(count / i)
    # x_ss_is.append(i)
ber_est=count / num_trials

print(ber_est,"SS-IS")
     
     
###because may be it finally could transmit, but with 2/3 transmissions, so why consider the other channel conditions where it could not, its enough the consider the last one, either it will be able to send or either it would not.