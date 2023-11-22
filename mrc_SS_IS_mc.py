import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm, rayleigh, expon,gaussian_kde
from scipy import special as sp
from numpy import asarray
from scipy import special as sp

db_16_ber = 1.73115703 * 10**(-5)
db_14_ber = 6.38754163 * 10**(-5)

# Simulation parameters
# this is a 1X3 MRC system
num_trials = 10000000  # Number of trials (bits transmitted per trial)

snr_db = 5
L = 3  #####Number of R_x
#########################
M = 2  #########BPSK constellation constant
m = np.arange(0, M)
constellation = 1 * np.cos(m / M * 2 * np.pi)
stream_0_1 = np.random.randint(low=0, high=M, size=num_trials)
ones_and_minus_ones = constellation[stream_0_1]
###################
mean_x = 0
var_x = 1
# mean_x_star = 0
# var_x_star = 1
noise1 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
noise2 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
noise3 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
# noise4 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
# noise5 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
# noise6 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
h1 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  ####channel coefficients
h2 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  ## channel coefficients
h3 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  #### channel coefficients
# h4 = norm.rvs(loc=mean_x, scale=var_x,
#               size=num_trials)  ####channel coefficients
# h5 = norm.rvs(loc=mean_x, scale=var_x,
#               size=num_trials)  ## channel coefficients
# h6 = norm.rvs(loc=mean_x, scale=var_x,
#               size=num_trials)  #### channel coefficients
# #################CSI at receiver
# H1=h1+h2*1j
# H2=h3+h4*1j
# H3=h5+h6*1j
# w1 = H1 / np.sqrt(abs(H1)**2 + abs(H2)**2 + abs(H3)**2)
# w2 = H2 / np.sqrt(abs(H1)**2 + abs(H2)**2 + abs(H3)**2)
# w3 = H3 / np.sqrt(abs(H1)**2 + abs(H2)**2 + abs(H3)**2)
w1 = np.conj(h1) / np.sqrt(np.absolute(h1)**2 + np.absolute(h2)**2 + np.absolute(h3)**2)
w2 = np.conj(h2) / np.sqrt(np.absolute(h1)**2 + np.absolute(h2)**2 + np.absolute(h3)**2)
w3 = np.conj(h3) / np.sqrt(np.absolute(h1)**2 + np.absolute(h2)**2 + np.absolute(h3)**2)
# #################################
P = sum(abs(ones_and_minus_ones)**2) / (num_trials)
snr = 10**(snr_db / 10)
th=1/snr
N0 = P / snr
# recv_1 = (H1 * ones_and_minus_ones) + (noise1+1j*noise2) * np.sqrt(N0 / 2)
# recv_2 = (H2 * ones_and_minus_ones) + (noise3+1j*noise4) * np.sqrt(N0 / 2)
# recv_3 = (H3 * ones_and_minus_ones) + (noise5+1j*noise6) * np.sqrt(N0 / 2)
recv_1 = (h1 * ones_and_minus_ones) + noise1 * np.sqrt(N0 / 2)
recv_2 = (h2 * ones_and_minus_ones) + noise2 * np.sqrt(N0 / 2)
recv_3 = (h3 * ones_and_minus_ones) + noise3 * np.sqrt(N0 / 2)
##############MRC receiver
dec_data = []
dec_data_zf = []
comb_at_recv = np.conj(w1) * recv_1 + np.conj(w2) * recv_2 + np.conj(w3) * recv_3
############################################ZF receiver
# nume=[]
# den=[]
# for i in range(num_trials):
#     den.append(h1[i]**2+h2[i]**2+h3[i]**2)
#     nume.append(h1[i]*recv_1[i]+h2[i]*recv_2[i]+h3[i]*recv_3[i])
# zf_recv=np.array(nume)/np.array(den)
# for i in zf_recv:
#     if i >0:
#         dec_data_zf.append(0)
#     elif i<0:
#         dec_data_zf.append(1)
########################################
#print(comb_at_recv[:100])
for i in comb_at_recv:
    if i >0:
        dec_data.append(0)
    else:
        dec_data.append(1)

####################SS_IS
l=th/0.9
a1 = np.log(1 + np.absolute(h1))
a2= np.log(1 + np.absolute(h2))
a3= np.log(1 + np.absolute(h3))
a4 = np.log(1 + np.absolute(noise1))
a5= np.log(1 + np.absolute(noise2))
a6= np.log(1 + np.absolute(noise3))
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
new_x1 = h1 * (th /l)**exp1  
new_x2 = h2 * (th /l)**exp2 
new_x3 = h3 * (th /l)**exp3 
new_x4 = noise1 * (th /l)**exp4 
new_x5 = noise2 * (th /l)**exp5 
new_x6 = noise3 * (th /l)**exp6 

weight1=(norm.pdf(new_x1,loc=np.mean(h1),scale=np.var(h1)))/(norm.pdf(new_x1,loc=np.mean(new_x1),scale=np.var(new_x1)))  #weight of 1st dim
weight2=(norm.pdf(new_x2,loc=np.mean(h2),scale=np.var(h2)))/(norm.pdf(new_x2,loc=np.mean(new_x2),scale=np.var(new_x2)))  #weight of 2nd dim
weight3=(norm.pdf(new_x3,loc=np.mean(h3),scale=np.var(h3)))/(norm.pdf(new_x3,loc=np.mean(new_x3),scale=np.var(new_x3)))  #weight of 3rd dim
weight4=(norm.pdf(new_x4,loc=np.mean(noise1),scale=np.var(noise1)))/(norm.pdf(new_x4,loc=np.mean(new_x4),scale=np.var(new_x4)))  #weight of 4th dim
weight5=(norm.pdf(new_x5,loc=np.mean(noise2),scale=np.var(noise2)))/(norm.pdf(new_x5,loc=np.mean(new_x5),scale=np.var(new_x5)))  #weight of 5th dim
weight6=(norm.pdf(new_x6,loc=np.mean(noise3),scale=np.var(noise3)))/(norm.pdf(new_x6,loc=np.mean(new_x6),scale=np.var(new_x6)))  #weight of 6th dim


# m1=np.mean(new_x1)
# m2=np.mean(new_x2)
# m3=np.mean(new_x3)

# v1=np.var(new_x1)
# v2=np.var(new_x2)
# v3=np.var(new_x3)
###############################################Sampling fn
# x_star = norm.rvs(loc=m1, scale=v1, size=num_trials)
# x_star2 = norm.rvs(loc=m2, scale=v2, size=num_trials)
# x_star3 = norm.rvs(loc=m3, scale=v3, size=num_trials)
############## SS-IS
ber_est = []
count = 0
count_mc=0
c_ss_is=[]
x_ss_is=[]
c_mc=[]
for i in range(1, len(stream_0_1)):
    # if (-1*(new_x1[i]+new_x4[i])+th>=th and -1*(new_x2[i]+new_x5[i])+th>=th and -1*(new_x3[i]+new_x6[i])+th>=th) or (np.absolute(new_x1[i])-new_x4[i]+th>=th and np.absolute(new_x2[i])-new_x5[i]+th>=th and np.absolute(new_x3[i])-new_x6[i]+th>=th) or (np.absolute(new_x4[i])-new_x1[i]+th>=th and np.absolute(new_x5[i])-new_x2[i]+th>=th and np.absolute(new_x6[i])-new_x3[i]+th>=th) or (np.absolute(new_x1[i])-np.absolute(new_x4[i])+th>=th and np.absolute(new_x2[i])-np.absolute(new_x5[i])+th>=th and np.absolute(new_x3[i])-np.absolute(new_x6[i])+th>=th) or ((-1*new_x1[i]+new_x4[i])+th>=th and (-1*new_x2[i]+new_x5[i])+th>=th and (-1*new_x3[i]+new_x6[i])+th>=th) or (new_x4[i]-new_x1[i]+th>=th and new_x5[i]-new_x2[i]+th>=th and new_x6[i]-new_x3[i]+th>=th):
    #     count = count + weight1[i]*weight2[i]*weight3[i]*weight4[i]*weight5[i]*weight6[i]
    # if (new_x1[i]<0 and new_x2[i]<0 and new_x3[i]<0 and new_x4[i]<0 and new_x5[i]<0 and new_x6[i]<0): #h -ve n-ve
        
    #     if  (np.absolute(new_x1[i])-np.absolute(new_x4[i])+th>=th and np.absolute(new_x2[i])-np.absolute(new_x5[i])+th>=th and np.absolute(new_x3[i])-np.absolute(new_x6[i])+th>=th):
    #         count = count + weight1[i]*weight2[i]*weight3[i]*weight4[i]*weight5[i]*weight6[i]
            
            
    # elif (new_x1[i]<0 and new_x2[i]<0 and new_x3[i]<0 and new_x4[i]>0 and new_x5[i]>0 and new_x6[i]>0): #h -ve n +ve
        
    #     if (np.absolute(new_x1[i])-new_x4[i]+th>=th and np.absolute(new_x2[i])-new_x5[i]+th>=th and np.absolute(new_x3[i])-new_x6[i]+th>=th):
    #         count = count + weight1[i]*weight2[i]*weight3[i]*weight4[i]*weight5[i]*weight6[i]
            
            
    # elif (new_x1[i]>0 and new_x2[i]>0 and new_x3[i]>0 and new_x4[i]<0 and new_x5[i]<0 and new_x6[i]<0): #h +ve n-ve
        
    #     if (np.absolute(new_x4[i])-new_x1[i]+th>th and np.absolute(new_x5[i])-new_x2[i]+th>th and np.absolute(new_x6[i])-new_x3[i]+th>th):
    #         count = count + weight1[i]*weight2[i]*weight3[i]*weight4[i]*weight5[i]*weight6[i]
            
            
    # elif (new_x1[i]>0 and new_x2[i]>0 and new_x3[i]>0 and new_x4[i]>0 and new_x5[i]>0 and new_x6[i]>0): #h +ve n +ve
        
    #     if (new_x4[i]-new_x1[i]+th>th and new_x5[i]-new_x2[i]+th>th and new_x6[i]-new_x3[i]+th>th):
    #         count = count + weight1[i]*weight2[i]*weight3[i]*weight4[i]*weight5[i]*weight6[i]
    
    
    if (w1[i]*new_x4[i])+(w2[i]*new_x5[i])+(w3[i]*new_x6[i])-np.sqrt((abs(new_x1[i])**2+abs(new_x2[i])**2+abs(new_x3[i])**2)*2*snr)+th>=th:
        count = count + weight1[i]*weight2[i]*weight3[i]*weight4[i]*weight5[i]*weight6[i]
        
    c_ss_is.append(count / i)
    x_ss_is.append(i)
ber_est.append(count /len(stream_0_1))

print(abs(ber_est[0]),"SS-IS")


for i in range(1, num_trials):
    if dec_data[i]!=stream_0_1[i] :
        count_mc = count_mc + 1
    c_mc.append(count_mc / i)
print(count_mc/num_trials,"van MC")

# for i in range(1, len(stream_0_1)):
#     if dec_data[i]!=stream_0_1[i] and abs(x_star[i])**2+abs(x_star2[i])**2+abs(x_star3[i])**2<=th:
#         count_mc = count_mc + ((norm.pdf(x_star[i], loc=np.mean(h1), scale=np.var(h1))/norm.pdf(x_star[i], loc=m1, scale=v1))*
#                                (norm.pdf(x_star2[i], loc=np.mean(h2), scale=np.var(h2))/norm.pdf(x_star2[i], loc=m2, scale=v2))*
#                                (norm.pdf(x_star3[i], loc=np.mean(h3), scale=np.var(h3))/norm.pdf(x_star3[i], loc=m3, scale=v3)))
#     c_mc.append(count_mc / (i))
# print(count_mc/(num_trials),"van IS")

# count_mc_2=0
# for i in range(0, len(stream_0_1)):
#     if dec_data_zf[i]!=stream_0_1[i] and abs(h1[i])**2+abs(h2[i])**2+abs(h3[i])**2<th:
#         count_mc_2 = count_mc_2 + 1
#     #c_mc.append(count_mc / i)
# print(count_mc_2/num_trials,"van MC 2")

# plt.plot(ones_and_minus_ones,comb_at_recv,'o')
# plt.hist(comb_at_recv,bins=100)
# # plt.hist(np.imag(comb_at_recv),bins=100)
# # # plt.plot(x_values2, estimated_pdf2, color='blue', label='Estimated PDF (KDE) err 2')
# # # plt.plot(x_values1,estimated_pdf,color="yellow",label='samp')
# # # #plt.legend()
# plt.grid(True)
# plt.show()

true = []
st = []
for j in range(num_trials):
    true.append(db_16_ber)
    st.append(j)

plt.figure()
plt.grid()
plt.plot(x_ss_is, c_ss_is, label="SS IS")
plt.plot(x_ss_is, c_mc, label="MC app")
plt.plot(st, true, label='True value',linestyle='dashed',color="gold",alpha=0.7)
plt.xlabel('Number of iterations')
plt.legend()
plt.show()
