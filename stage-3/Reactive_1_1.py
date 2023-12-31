#####same packet will be sent in HARQ retx
#####collision of packets is not considerd as a event where HARQ retx is extinguised
#####Grant Access Failure: In M HARQ tx the UE is unable to take access of the BS

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm, rayleigh, expon,randint
from scipy import special as sp
from numpy import asarray


MHT=3  ###maximum HARQ repetitions
num_trials=500000
MNU=4  ####maximum number of users+1
snr_db = 2
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
X_mc=[]
Y_mc=[]
M_cos=[]
H2=[]
H3=[]
W1=[]
W2=[]
W3=[]
n1=[]
n2=[]
n3=[]
for a in range(1,num_trials):
    M_co=0 #########this is the termination
    while M_co<MHT:
      
        mean_x1=0
        var_x1 = 1
        
        noise1 = norm.rvs(loc=mean_x1, scale=var_x1, size=1)
        noise2 = norm.rvs(loc=mean_x1, scale=var_x1, size=1)
        noise3 = norm.rvs(loc=mean_x1, scale=var_x1, size=1)
        h1 = norm.rvs(loc=mean_x1, scale=var_x1,
                    size=1)  ####channel coefficients
        h2 = norm.rvs(loc=mean_x1, scale=var_x1,
                        size=1)  ## channel coefficients
        h3 = norm.rvs(loc=mean_x1, scale=var_x1,
                        size=1)  #### channel coefficients
        #########BPSK constellation constant
        M = 2  
        m = np.arange(0, M)
        constellation = 1 * np.cos(m / M * 2 * np.pi)
        stream_0_1 = np.random.randint(low=0, high=M, size=1)    
        ones_and_minus_ones = constellation[stream_0_1]

        w1 = np.conj(h1) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
        w2 = np.conj(h2) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
        w3 = np.conj(h3) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
        
        H1.append(h1[0])
        H2.append(h2[0])
        H3.append(h3[0])
        W1.append(w1[0])
        W2.append(w2[0])
        W3.append(w3[0])
        n1.append(noise1[0])
        n2.append(noise2[0])
        n3.append(noise3[0])
        
        ###########################################
       
        ##########################
        P = 1
        
        N0 = P / snr
       
        recv_1 = (h1 * ones_and_minus_ones) + noise1 * np.sqrt(N0 / 2)
        recv_2 = (h2 * ones_and_minus_ones) + noise2 * np.sqrt(N0 / 2)
        recv_3 = (h3 * ones_and_minus_ones) + noise3 * np.sqrt(N0 / 2)
        comb_at_recv = np.conj(w1) * recv_1 + np.conj(w2) * recv_2 + np.conj(w3) * recv_3
        dec=0
        for i in comb_at_recv:
            if (i >0):
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
    # I.append(i[0])
    # J.append(j[0])
    #print(M_co)
###MONTE CARLO METHOD           
    if M_co==MHT:  ######Grant access failure: how many times in num_trials M_co>4    
        #print(M_co)    
        gf=gf+1
    Y_mc.append(gf/a)
    X_mc.append(a)
print(gf/num_trials)
#print(c)
print(len(H1)) 

  
l=th/3
a1= np.log(1 + np.absolute(H1))
a2= np.log(1 + np.absolute(H2))
a3= np.log(1 + np.absolute(H3))
a4= np.log(1 + np.absolute(n1))
a5= np.log(1 + np.absolute(n2))
a6= np.log(1 + np.absolute(n3))
#b=np.log(l)
b1=np.sqrt(np.sum(np.absolute(a1**2)))
b2=np.sqrt(np.sum(np.absolute(a2**2)))
b3=np.sqrt(np.sum(np.absolute(a3**2)))
b4=np.sqrt(np.sum(np.absolute(a4**2)))
b5=np.sqrt(np.sum(np.absolute(a5**2)))
b6=np.sqrt(np.sum(np.absolute(a6**2)))
#print(b)
exp1=a1/(b1)
exp2=a2/(b2)
exp3=a3/(b3)
exp4=a4/(b4)
exp5=a5/(b5)
exp6=a6/(b6)
new_x1 = H1 * (th /l)**exp1  
new_x2 = H2 * (th /l)**exp2 
new_x3 = H3 * (th /l)**exp3 
new_x4 = n1 * (th /l)**exp4 
new_x5 = n2 * (th /l)**exp5 
new_x6 = n3 * (th /l)**exp6 


weight1=(norm.pdf(new_x1,loc=np.mean(H1),scale=np.var(H1)))/(norm.pdf(new_x1,loc=np.mean(new_x1),scale=np.var(new_x1)))  #weight of 1st dim
weight2=(norm.pdf(new_x2,loc=np.mean(H2),scale=np.var(H2)))/(norm.pdf(new_x2,loc=np.mean(new_x2),scale=np.var(new_x2)))  #weight of 2nd dim
weight3=(norm.pdf(new_x3,loc=np.mean(H3),scale=np.var(H3)))/(norm.pdf(new_x3,loc=np.mean(new_x3),scale=np.var(new_x3)))  #weight of 3rd dim
weight4=(norm.pdf(new_x4,loc=np.mean(n1),scale=np.var(n1)))/(norm.pdf(new_x4,loc=np.mean(new_x4),scale=np.var(new_x4)))  #weight of 4th dim
weight5=(norm.pdf(new_x5,loc=np.mean(n2),scale=np.var(n2)))/(norm.pdf(new_x5,loc=np.mean(new_x5),scale=np.var(new_x5)))  #weight of 5th dim
weight6=(norm.pdf(new_x6,loc=np.mean(n3),scale=np.var(n3)))/(norm.pdf(new_x6,loc=np.mean(new_x6),scale=np.var(new_x6)))  #weight of 6th dim
# there will be two more weights.the problem params will be abs(new_x1[i])**2+abs(new_x2[i])**2+abs(new_x3[i])**2<th and new_i==new_j
#print(weight2)
##############SS Importance Sampling
count = 0
i=0
y_ss_is=[]
x_ss_is=[]
#print(len(new_x1))
# for i in range(1, len(H1)):
#for i in range(3, len(H1)-2,k):
while i<len(H1)-2:
    if (W1[i]*new_x4[i])+(W2[i]*new_x5[i])+(W3[i]*new_x6[i])-np.sqrt((abs(new_x1[i])**2+abs(new_x2[i])**2+abs(new_x3[i])**2)*2*snr)+th>=th:
        i=i+1
        if (W1[i]*new_x4[i])+(W2[i]*new_x5[i])+(W3[i]*new_x6[i])-np.sqrt((abs(new_x1[i])**2+abs(new_x2[i])**2+abs(new_x3[i])**2)*2*snr)+th>=th:
            i=i+1
            if (W1[i]*new_x4[i])+(W2[i]*new_x5[i])+(W3[i]*new_x6[i])-np.sqrt((abs(new_x1[i])**2+abs(new_x2[i])**2+abs(new_x3[i])**2)*2*snr)+th>=th:
                count = count + weight1[i]*weight2[i]*weight3[i]*weight4[i]*weight5[i]*weight6[i]
    # 
    i=i+1

    # if (new_x1[i]<0 and new_x2[i]<0):
    #     if  (np.absolute(new_x1[i])-np.absolute(new_x2[i])+th>=th):
    #         count = count + weight1[i]*weight2[i]
    # elif (new_x1[i]<0 and new_x2[i]>0 ):
    #     if (np.absolute(new_x1[i])-new_x2[i]+th>=th) :
    #         count = count + weight1[i]*weight2[i]
    # elif (new_x1[i]>0 and new_x2[i]<0):
    #     if (np.absolute(new_x2[i])-new_x1[i]+th>=th):
    #         count = count + weight1[i]*weight2[i]
    # elif (new_x1[i]>0 and new_x2[i]>0):
    #     if (new_x2[i]-new_x1[i]+th>=th):
    #         count = count + weight1[i]*weight2[i]
# ber_est=count / len(H1)
ber_est=count / num_trials
#print(ber_est,"hi")
print(ber_est,"SS-IS")


# plt.figure()
# plt.grid()
# plt.plot(X_mc, Y_mc, label="MC")
# # plt.plot(x_mc, c_is, label="IS")
# plt.plot(x_ss_is, y_ss_is, label="SS IS",color="black",linestyle='dashed')
# # plt.plot(st, true, label='True value',linestyle='dashed',color="gold",alpha=0.7)
# plt.xlabel('Number of iterations')
# plt.legend()
# plt.show()
     
###because may be it finally could transmit, but with 2/3 transmissions, so why consider the other channel conditions where it could not, its enough the consider the last one, either it will be able to send or either it would not.