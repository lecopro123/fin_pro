#####same packet will be sent in HARQ retx
#####collision of packets is not considerd as a event where HARQ retx is extinguised
#####Grant Access Failure: In M HARQ tx the UE is unable to take access of the BS

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm, rayleigh, expon,randint
from scipy import special as sp
from numpy import asarray
import random

MHT=3  ###maximum HARQ repetitions
num_trials=100000
MNU=4  ####maximum number of users+1
snr_db = 9
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
#print(th/2)
H1=[]
N1=[]
X_mc=[]
Y_mc=[]
M_cos=[]
ch=[]
J=[]
def checker(arr):
    max=arr[0]
    pl=0
    for i in range(len(arr)):
        if arr[i]>max:
            max=arr[i]
            pl=i
    if max>1:
        return max
    else:
        return 0
# def checker2(arr):
#     max=arr[0]
#     #pl=0
#     for i in range(len(arr)):
#         if arr[i]>max:
#             max=arr[i]
#             #pl=i
#     if max>1:
#         return -1
#     else:
#         return th*max
for a in range(1,num_trials):
    M_co=0 #########this is the termination
    while M_co<MHT:
        #if(M_co==4): print("Hi")
        choice=norm.rvs(loc=0,scale=1,size=3)
        mean_x1=0
        var_x1 = 1
        # mean_x2=2
        # var_x2= 0.5
        # mean_x3=0
        # var_x3 = 1.5
        noise1 = norm.rvs(loc=mean_x1, scale=var_x1, size=1)
        # noise2 = norm.rvs(loc=mean_x1, scale=var_x1, size=1)
        # noise3 = norm.rvs(loc=mean_x2, scale=var_x2, size=1)
        # noise4 = norm.rvs(loc=mean_x2, scale=var_x2, size=1)
        # noise5 = norm.rvs(loc=mean_x3, scale=var_x3, size=1)
        # noise6 = norm.rvs(loc=mean_x3, scale=var_x3, size=1)
        h1 = norm.rvs(loc=mean_x1, scale=var_x1,
                    size=1)  ####channel coefficients
        # h2 = norm.rvs(loc=mean_x1, scale=var_x1,
        #             size=1)  ## channel coefficients
        # h3 = norm.rvs(loc=mean_x2, scale=var_x2,
        #             size=1)  #### channel coefficients
        #print(i,j)
        H1.append(h1[0])
        N1.append(noise1[0])
        # if checker(choice)>1:
        ch.append(checker(choice))
        # else:
        #     ch.append(random.random())
        if (not checker(choice)):
            #print("User:" )
            M_co=M_co+1
        else:
            #M_co=M_co+1
            #########BPSK constellation constant
            M = 2  
            m = np.arange(0, M)
            constellation = 1 * np.cos(m / M * 2 * np.pi)
            stream_0_1 = np.random.randint(low=0, high=M, size=1)    ##########this steps ensure that the max Harq retransmissions are 4.
            ones_and_minus_ones = constellation[stream_0_1]
            ###########################################
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

            # w1 = np.conj(h1) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
            # w2 = np.conj(h2) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
            # w3 = np.conj(h3) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
            ##########################
            P = 1
            # snr = 10**(snr_db / 10)
            N0 = P / snr
            # recv_1 = (H1 * ones_and_minus_ones) + (noise1+1j*noise2) * np.sqrt(N0 / 2)
            # recv_2 = (H2 * ones_and_minus_ones) + (noise3+1j*noise4) * np.sqrt(N0 / 2)
            # recv_3 = (H3 * ones_and_minus_ones) + (noise5+1j*noise6) * np.sqrt(N0 / 2)
            recv_1 = (h1[0] * ones_and_minus_ones[0]) + noise1[0] * np.sqrt(N0 / 2)
            # recv_2 = (h2 * ones_and_minus_ones) + noise2 * np.sqrt(N0 / 2)
            # recv_3 = (h3 * ones_and_minus_ones) + noise3 * np.sqrt(N0 / 2)
            #########################Combination
            #comb_at_recv = np.conj(w1) * recv_1 + np.conj(w2) * recv_2 + np.conj(w3) * recv_3
            # print(comb_at_recv)
            # print(ones_and_minus_ones)
            dec=0
            #print(recv_1)
            #for i in recv_1:#comb_at_recv:
            if (recv_1 >0):
                dec=0
            else:
                dec=1
            ###############################
            if dec != stream_0_1: ############Counter M++
                M_co = M_co + 1
            # if noise1[0]>abs(h1[0])*np.sqrt(2*snr): ############Counter M++
            #     M_co = M_co + 1
            else:    ###########conn established
                break
    
    #print(M_co)
###MONTE CARLO METHOD           
    if M_co==MHT:  ######Grant access failure: how many times in num_trials M_co>4    
        #print(M_co)    
        gf=gf+1
    Y_mc.append(gf/a)
    X_mc.append(a)
print(gf/num_trials)
#print(c)
#print(len(H1))   

l=th/8
a1 = np.log(1 + np.absolute(H1))
a2= np.log(1 + np.absolute(ch))
a4= np.log(1 + np.absolute(N1))
# b=np.log(l)
b1=np.sqrt(np.sum(np.absolute(a1**2)))
b2=np.sqrt(np.sum(np.absolute(a2**2)))
b4=np.sqrt(np.sum(np.absolute(a4**2)))
exp4=a4/b4
exp2=a2/b2
exp1=a1/b1
new_x1 = H1 * (th /l)**exp1  
new_x2 = ch * (th /l)**exp2 
new_x4 = N1 * (th/l)**exp4 


weight1=(norm.pdf(new_x1,loc=np.mean(H1),scale=np.var(H1)))/(norm.pdf(new_x1,loc=np.mean(new_x1),scale=np.var(new_x1)))  #weight of 1st dim
weight4=(norm.pdf(new_x4,loc=np.mean(N1),scale=np.var(N1)))/(norm.pdf(new_x4,loc=np.mean(new_x4),scale=np.var(new_x4)))  #weight of 4st dim
weight2=(norm.pdf(new_x2,loc=np.mean(ch),scale=np.var(ch)))/(norm.pdf(new_x2,loc=np.mean(new_x2),scale=np.var(new_x2)))  #weight of 2nd dim

# there will be two more weights.the problem params will be abs(new_x1[i])**2+abs(new_x2[i])**2+abs(new_x3[i])**2<th and new_i==new_j
#print(weight2)
##############SS Importance Sampling
count = 0
i=0
y_ss_is=[]
x_ss_is=[]
#print(len(new_x1))
# for i in range(1, len(H1)):
while i<len(H1)-2:
    # if (W1[i]*new_x4[i])+(W2[i]*new_x5[i])+(W3[i]*new_x6[i])-np.sqrt((abs(new_x1[i])**2+abs(new_x2[i])**2+abs(new_x3[i])**2)*2*snr)+th>=th and (W1[i+1]*new_x4[i+1])+(W2[i+1]*new_x5[i+1])+(W3[i+1]*new_x6[i+1])-np.sqrt((abs(new_x1[i+1])**2+abs(new_x2[i+1])**2+abs(new_x3[i+1])**2)*2*snr)+th>=th and (W1[i+2]*new_x4[i+2])+(W2[i+2]*new_x5[i+2])+(W3[i+2]*new_x6[i+2])-np.sqrt((abs(new_x1[i+2])**2+abs(new_x2[i+2])**2+abs(new_x3[i+2])**2)*2*snr)+th>=th:
    #     count = count + weight1[i]*weight2[i]*weight3[i]*weight4[i]*weight5[i]*weight6[i]
    # c_ss_is.append(count / i)
    # x_ss_is.append(i)
    if (new_x2[i])>1 or (new_x4[i])-np.sqrt((abs(new_x1[i])**2)*2*snr)+th>=th:
        i=i+1
        if (new_x2[i])>1 or (new_x4[i])-np.sqrt((abs(new_x1[i])**2)*2*snr)+th>=th:
            i=i+1
            if (new_x2[i])>1 or (new_x4[i])-np.sqrt((abs(new_x1[i])**2)*2*snr)+th>=th:
                count = count + weight1[i]*weight2[i]*weight4[i]
    i=i+1
#ber_est=count / len(H1)
ber_est=count / num_trials
#print(ber_est,"hi")
print(ber_est,"SS-IS")


plt.figure()
plt.grid()
plt.plot(X_mc, Y_mc, label="MC")
# plt.plot(x_mc, c_is, label="IS")
plt.plot(x_ss_is, y_ss_is, label="SS IS",color="black",linestyle='dashed')
# plt.plot(st, true, label='True value',linestyle='dashed',color="gold",alpha=0.7)
plt.xlabel('Number of iterations')
plt.legend()
plt.show()
     
###because may be it finally could transmit, but with 2/3 transmissions, so why consider the other channel conditions where it could not, its enough the consider the last one, either it will be able to send or either it would not.