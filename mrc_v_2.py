import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm, rayleigh, expon,gaussian_kde
from scipy import special as sp
from numpy import asarray


db_16_ber = 1.73115703 * 10**(-5)
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
var_x = 1.1
mean_x_star = 2
var_x_star = 1
noise1 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
noise2 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
noise3 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
noise4 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
noise5 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
noise6 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
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
x_star = norm.rvs(loc=mean_x_star, scale=var_x_star, size=num_trials)
x_star2 = norm.rvs(loc=mean_x_star, scale=var_x_star, size=num_trials)
x_star3 = norm.rvs(loc=mean_x_star, scale=var_x_star, size=num_trials)
x_star4 = norm.rvs(loc=mean_x_star, scale=var_x_star, size=num_trials)
x_star5 = norm.rvs(loc=mean_x_star, scale=var_x_star, size=num_trials)
x_star6 = norm.rvs(loc=mean_x_star, scale=var_x_star, size=num_trials)
samp_1=x_star+1j*x_star2
samp_2=x_star3+1j*x_star4
samp_3=x_star5+1j*x_star6
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




# err_s_1=[]
# err_s_0=[]
# for i in range(len(comb_at_recv)):
#     if(dec_data[i]==0 and stream_0_1[i]==1):
#         err_s_1.append(np.real(comb_at_recv[i]))
#     elif(dec_data[i]==1 and stream_0_1[i]==0):
#         err_s_0.append(np.real(comb_at_recv[i]))
# print(comb_at_recv[0])
# x_values1 = np.linspace(min(np.real(comb_at_recv)), max(np.real(comb_at_recv)), num_trials)


# # Estimate the kernel density function
# kde1 = gaussian_kde(np.real(comb_at_recv))


# # Evaluate the KDE at the x values
# estimated_pdf = kde1.evaluate(x_values1)
# normalized_pdf = estimated_pdf / estimated_pdf.sum()


# x_values2 = np.linspace(min(err_s_0), max(err_s_0), num_trials)


# # Estimate the kernel density function
# kde2 = gaussian_kde(err_s_0)
# # Evaluate the KDE at the x values
# estimated_pdf2 = kde2.evaluate(x_values2)
# #print(np.mean(x_values1))
# normalized_pdf2 = estimated_pdf2 / estimated_pdf2.sum()




#print(np.var(x_values2))


# x_star = norm.rvs(loc=np.mean(x_values2), scale=np.var(x_values1), size=num_trials)
# x_star_2 = norm.rvs(loc=np.mean(x_values1), scale=np.var(x_values1), size=num_trials)
# print(kde2.pdf(x_star)[:100])


# hist_values, bin_edges = np.histogram(comb_at_recv, bins=100000, density=True)


# Calculate bin widths
# bin_widths = bin_edges[1:] - bin_edges[:-1]


# # Convert histogram to PDF by normalizing by bin width and total number of data points
# pdf = hist_values / (hist_values.sum() * bin_widths)
##############Importance Sampling
ber_est = []
count = 0
#count_mc = 0
#count_bpsk = 0
# for j in range(100):
#     count = 0
for i in range(0, len(stream_0_1)):
    if dec_data[i] != stream_0_1[i] and (abs(H1[i])**2+ abs(H2[i])**2+ abs(H3[i])**2 <1/np.sqrt(snr)):#and (abs(x_star[i]) < 1/np.sqrt(snr) and abs(x_star2[i]) < 1/np.sqrt(snr) and abs(x_star2[i]) < 1/np.sqrt(snr)):
        count = count + 1
ber_est.append(count / num_trials)


print(ber_est)










# plt.plot(ones_and_minus_ones,comb_at_recv,'o')
# x = [ele.real for ele in comb_at_recv]
# # extract imaginary part
# y = [ele.imag for ele in comb_at_recv]
 
# # plot the complex numbers
# plt.scatter(x, y)
# #plt.hist(np.real(comb_at_recv),bins=100)
# #plt.hist(np.imag(comb_at_recv),bins=100)
# #plt.plot(x_values2, estimated_pdf2, color='blue', label='Estimated PDF (KDE) err 2')
# #plt.plot(x_values1,estimated_pdf,color="yellow",label='samp')
# # #plt.legend()
# plt.grid(True)
# plt.show()




# plt.plot(x_star, kde2.pdf(x_star), color='blue', label='Estimated PDF (KDE) err 2')
# plt.legend()
# plt.grid(True)
# plt.show()



