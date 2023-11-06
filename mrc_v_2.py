import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm, rayleigh, expon,gaussian_kde
from scipy import special as sp
from numpy import asarray


db_16_ber = 1.747 * 10**(-5)
db_14_ber = 6.38754163 * 10**(-5)


# Simulation parameters
# this is a 1X3 MRC system
num_trials = 10000000  # Number of trials (bits transmitted per trial)
#sims=1000 # Number of simulations
# snr_range_db = np.arange(-10, 18, 2)  # SNR range in dB
# print(snr_range_db)
# snr_range = 10**(snr_range_db / 10)  ####SNR raw values
snr_db = 16
L = 3  #####Number of R_x
#########################
M = 2  #########BPSK constellation constant
m = np.arange(0, M)
constellation = 1 * np.cos(m / M * 2 * np.pi)
stream_0_1 = np.random.randint(low=0, high=M, size=num_trials)
ones_and_minus_ones = constellation[stream_0_1]
###################
mean_x = 0
var_x = 1.4
var_xn=1.7
mean_x_star = 2
var_x_star = 1
noise1 = norm.rvs(loc=mean_x, scale=var_xn, size=num_trials)
noise2 = norm.rvs(loc=mean_x, scale=var_xn, size=num_trials)
noise3 = norm.rvs(loc=mean_x, scale=var_xn, size=num_trials)
noise4 = norm.rvs(loc=mean_x, scale=var_xn, size=num_trials)
noise5 = norm.rvs(loc=mean_x, scale=var_xn, size=num_trials)
noise6 = norm.rvs(loc=mean_x, scale=var_xn, size=num_trials)
h1 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  ####channel coefficients
h2 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  ## channel coefficients
h3 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  #### channel coefficients
h4 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  ####channel coefficients
h5 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  ## channel coefficients
h6 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  #### channel coefficients
#print(ones_and_minus_ones)
# x_star = norm.rvs(loc=mean_x_star, scale=var_x_star, size=num_trials)
# x_star2 = norm.rvs(loc=mean_x_star, scale=var_x_star, size=num_trials)
# x_star3 = norm.rvs(loc=mean_x_star, scale=var_x_star, size=num_trials)
# x_star4 = norm.rvs(loc=mean_x_star, scale=var_x_star, size=num_trials)
# x_star5 = norm.rvs(loc=mean_x_star, scale=var_x_star, size=num_trials)
# x_star6 = norm.rvs(loc=mean_x_star, scale=var_x_star, size=num_trials)
# samp_1=x_star+1j*x_star2
# samp_2=x_star3+1j*x_star4
# samp_3=x_star5+1j*x_star6
# #################CSI at receiver
H1=h1+h2*1j
H2=h3+h4*1j
H3=h5+h6*1j
w1 = H1 / np.sqrt(abs(H1)**2 + abs(H2)**2 + abs(H3)**2)
w2 = H2 / np.sqrt(abs(H1)**2 + abs(H2)**2 + abs(H3)**2)
w3 = H3 / np.sqrt(abs(H1)**2 + abs(H2)**2 + abs(H3)**2)
# w1 = np.conj(h1) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
# w2 = np.conj(h2) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
# w3 = np.conj(h3) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
# #################################
P = sum(abs(ones_and_minus_ones)**2) / (num_trials)
snr = 10**(snr_db / 10)
N0 = P / snr
recv_1 = (np.sqrt(1/2)*H1 * ones_and_minus_ones) + (noise1+1j*noise2) * np.sqrt(N0 / 2)
recv_2 = (np.sqrt(1/2)*H2 * ones_and_minus_ones) + (noise1+1j*noise2) * np.sqrt(N0 / 2)
recv_3 = (np.sqrt(1/2)*H3 * ones_and_minus_ones) + (noise1+1j*noise2) * np.sqrt(N0 / 2)
# recv_1 = (h1 * ones_and_minus_ones) + noise1 * np.sqrt(N0 / 2)
# recv_2 = (h2 * ones_and_minus_ones) + noise2 * np.sqrt(N0 / 2)
# recv_3 = (h3 * ones_and_minus_ones) + noise3 * np.sqrt(N0 / 2)
##############receiver
dec_data = []
dec_data_bpsk = []
comb_at_recv = np.conj(w1) * recv_1 + np.conj(w2) * recv_2 + np.conj(w3) * recv_3
#print(comb_at_recv[:100])
for i in comb_at_recv:
    if (np.real(i) >0):
        dec_data.append(0)
    #elif (np.real(i) < 0 and np.imag(i) < 0):
    elif np.real(i)<=0:
        dec_data.append(1)


##############MC
c_mc = []
x_mc=[]
count = 0
#count_mc = 0
#count_bpsk = 0
# for j in range(100):
#     count = 0
for i in range(1, len(stream_0_1)):
    if dec_data[i] != stream_0_1[i] :#and (abs(H1[i])**2+ abs(H2[i])**2+ abs(H3[i])**2 <1/np.sqrt(snr)):#and (abs(x_star[i]) < 1/np.sqrt(snr) and abs(x_star2[i]) < 1/np.sqrt(snr) and abs(x_star2[i]) < 1/np.sqrt(snr)):
        count = count + 1
    c_mc.append(count / i)
    x_mc.append(i)
print(count / num_trials,"Van MC")


###################SS-IS
mean_x = 0
var_x = 2
noise1 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
noise2 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
noise3 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
h1 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  ####channel coefficients
h2 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  ## channel coefficients
h3 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  #### channel coefficients
w1 = np.conj(h1) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
w2 = np.conj(h2) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
w3 = np.conj(h3) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
# #################################
P = sum(abs(ones_and_minus_ones)**2) / (num_trials)
snr = 10**(snr_db / 10)
th=1/np.sqrt(snr)
N0 = P / snr
recv_1 = (h1 * ones_and_minus_ones) + noise1 * np.sqrt(N0 / 2)
recv_2 = (h2 * ones_and_minus_ones) + noise2 * np.sqrt(N0 / 2)
recv_3 = (h3 * ones_and_minus_ones) + noise3 * np.sqrt(N0 / 2)
##############MRC receiver
dec_data = []
comb_at_recv = np.conj(w1) * recv_1 + np.conj(w2) * recv_2 + np.conj(w3) * recv_3

########################################
#print(comb_at_recv[:100])
for i in comb_at_recv:
    if i >0:
        dec_data.append(0)
    elif i<0:
        dec_data.append(1)

####################SS_IS
a1 = np.log(1 + np.absolute(h1))
a2= np.log(1 + np.absolute(h2))
a3= np.log(1 + np.absolute(h3))
b=np.log(9.9)
exp3=a3/b
exp2=a2/b
exp1=a1/b
new_x1 = h1 * (th /0.9)**exp1  
new_x2 = h2 * (th /0.7)**exp2 
new_x3 = h3 * (th /0.9)**exp3 


weight1=(norm.pdf(new_x1,loc=np.mean(h1),scale=np.var(h1)))/(norm.pdf(new_x1,loc=np.mean(new_x1),scale=np.var(new_x1)))  #weight of 1st dim
weight2=(norm.pdf(new_x2,loc=np.mean(h2),scale=np.var(h2)))/(norm.pdf(new_x2,loc=np.mean(new_x2),scale=np.var(new_x2)))  #weight of 2nd dim
weight3=(norm.pdf(new_x3,loc=np.mean(h3),scale=np.var(h3)))/(norm.pdf(new_x3,loc=np.mean(new_x3),scale=np.var(new_x3)))  #weight of 3rd dim

##############SS Importance Sampling
ber_est = []
count = 0

c_ss_is=[]
x_ss_is=[]

for i in range(1, len(stream_0_1)):
    if abs(new_x1[i])**2+abs(new_x2[i])**2+abs(new_x3[i])**2<th:
        count = count + weight1[i]*weight2[i]*weight3[i]
    c_ss_is.append(count / i)
    x_ss_is.append(i)
ber_est.append(count / num_trials)

print(abs(ber_est[0]),"SS-IS")


######################################IS
mean_x = 0
var_x = 0.9
var_xn=1.9
noise1 = norm.rvs(loc=mean_x, scale=var_xn, size=num_trials)
noise2 = norm.rvs(loc=mean_x, scale=var_xn, size=num_trials)
noise3 = norm.rvs(loc=mean_x, scale=var_xn, size=num_trials)
h1 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  ####channel coefficients
h2 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  ## channel coefficients
h3 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  #### channel coefficients
w1 = np.conj(h1) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
w2 = np.conj(h2) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
w3 = np.conj(h3) / np.sqrt(abs(h1)**2 + abs(h2)**2 + abs(h3)**2)
# #################################
P = sum(abs(ones_and_minus_ones)**2) / (num_trials)
snr = 10**(snr_db / 10)
th=1/np.sqrt(snr)
N0 = P / snr
recv_1 = (h1 * ones_and_minus_ones) + noise1 * np.sqrt(N0 / 2)
recv_2 = (h2 * ones_and_minus_ones) + noise2 * np.sqrt(N0 / 2)
recv_3 = (h3 * ones_and_minus_ones) + noise3 * np.sqrt(N0 / 2)
##############MRC receiver
dec_data = []
comb_at_recv = np.conj(w1) * recv_1 + np.conj(w2) * recv_2 + np.conj(w3) * recv_3

########################################
#print(comb_at_recv[:100])
for i in comb_at_recv:
    if i >0:
        dec_data.append(0)
    elif i<0:
        dec_data.append(1)
m1=np.mean(new_x1)
m2=np.mean(new_x2)
m3=np.mean(new_x3)

v1=np.var(new_x1)
v2=np.var(new_x2)
v3=np.var(new_x3)
print(m1,v1)
print(m2,v2)
print(m3,v3)
###############################################Sampling fn
x_star = norm.rvs(loc=m1, scale=v1, size=num_trials)
x_star2 = norm.rvs(loc=m2, scale=v2, size=num_trials)
x_star3 = norm.rvs(loc=m3, scale=v3, size=num_trials)

count_is=0
for i in range(1, len(stream_0_1)):
    if dec_data[i] != stream_0_1[i] and  abs(x_star[i])**2+abs(x_star2[i])**2+abs(x_star3[i])**2<=1/np.sqrt(snr):
        count_is = count_is + ((norm.pdf(x_star[i], loc=np.mean(h1), scale=np.var(h1))/norm.pdf(x_star[i]))*
                            (norm.pdf(x_star2[i], loc=np.mean(h2), scale=np.var(h2))/norm.pdf(x_star2[i])))
                            #    (norm.pdf(x_star3[i], loc=np.mean(h3), scale=np.var(h3))/norm.pdf(x_star3[i])))
    #c_mc.append(count_mc / (i))
print(count_is/num_trials,"van IS")


# true = []
# st = []