########taking number of antennas L=3 
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm,rayleigh,expon,gaussian_kde
from scipy import special as sp
from numpy import asarray
# Simulation parameters
# this is a 1X3 MRC system
num_trials = 10000  # Number of trials (bits transmitted per trial)
#sims=1000 # Number of simulations
snr_range_db = np.arange(-10, 11, 2)  # SNR range in dB
snr_range = 10**(snr_range_db / 10)  ####SNR raw values
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
mean_x_star =0
var_x_star = 3
noise1 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
noise2 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
noise3 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
h1 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  ####channel coefficients
h2 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  ## channel coefficients
h3 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  #### channel coefficients
#print(ones_and_minus_ones)
x_star = norm.rvs(loc=1.1, scale=0.3, size=num_trials)
x_star2 = norm.rvs(loc=-1.1, scale=0.3, size=num_trials)
x_star3 = norm.rvs(loc=1.1, scale=4, size=num_trials)
x_star1 = norm.rvs(loc=mean_x_star, scale=var_x_star, size=num_trials)

mean1 = -1.1
std1 = 0.6
mean2 = 1.1
std2 = 0.6

# Create a range of x values
x = np.linspace(-3, 3, 100000)

# Compute the PDFs of the two Gaussian distributions
pdf1 = norm.pdf(x, mean1, std1)
pdf2 = norm.pdf(x, mean2, std2)

# Combine the PDFs to create the custom distribution
custom_pdf = pdf1 + pdf2

# Normalize the custom PDF
custom_pdf /= custom_pdf.sum()

###################################### theoritical BER
lam = np.sqrt(snr_range / (2 + snr_range))  ######a constant in MRC BER formula
ber_theo_coef = ((1 - lam) / 2)**L
ber_coef_theo_sum = 0
for i in range(L):
    ber_coef_theo_sum = ber_coef_theo_sum + (
        (math.factorial(L + i - 1) /
         (math.factorial(L - 1) * math.factorial(i))) * (0.5 * (1 + lam))**2)
#ber_theoretical_mrc = 0.5 * (1 - np.sqrt(1 / (1 + 10**(snr_range_db / 10))))

# def q_function(x):
#     return 0.5 * (1 - sp.erf(x / np.sqrt(2)))

# a = 0
# for i in range(0, num_trials):
#     a = a + np.abs(h1[i])**2 + np.abs(h2[i])**2 + np.abs(h3[i])**2
# ber_t = []
# for i in snr_range:
#     ber_t.append(q_function(np.sqrt(i * np.random.chisquare(3, 1)[0])))
#print(ber_coef_theo_sum * ber_theo_coef)

#################CSI at receiver
w1 = np.conj(h1 / np.sqrt(np.abs(h1) + np.abs(h2) + np.abs(h3)))
w2 = np.conj(h2 / np.sqrt(np.abs(h1) + np.abs(h2) + np.abs(h3)))
w3 = np.conj(h3 / np.sqrt(np.abs(h1) + np.abs(h2) + np.abs(h3)))
#################################

# sim_ber_est=[]
# sim_ber_est_mc=[]
# for j in range(1,sims+1):
ber_est = []
ber_est_mc = []
#ber_est_bpsk = []
for i in snr_range:
    P = sum(abs(ones_and_minus_ones)**2) / (num_trials)
    #print(P)
    N0 = P / i
    recv_1 = (h1 * ones_and_minus_ones) + noise1 * np.sqrt(N0 / 2)
    recv_2 = (h2 * ones_and_minus_ones) + noise2 * np.sqrt(N0 / 2)
    recv_3 = (h3 * ones_and_minus_ones) + noise3 * np.sqrt(N0 / 2)
    snr=i
    ##############receiver
    dec_data = []
    dec_data_bpsk = []
    comb_at_recv = w1 * recv_1 + w2 * recv_2 + w3 * recv_3
    x = np.linspace(min(comb_at_recv) - 1, max(comb_at_recv) + 1, num_trials)
    kde = gaussian_kde(comb_at_recv)
    pdf_estimated = kde(x)
    
    for i in comb_at_recv:
        if (i > 0):
            dec_data.append(0)
        else:
            dec_data.append(1)

    # for i in recv_1:
    #     if (i > 0):
    #         dec_data_bpsk.append(0)
    #     else:
    #         dec_data_bpsk.append(1)

##############Importance Sampling
    count = 0
    count_mc = 0
    #count_bpsk = 0
    for i in range(0, len(stream_0_1)):
          if dec_data[i]==0 and stream_0_1[i]==1 and x_star[i]<0 :
            count_mc = count_mc + 1
            count = count + (
                (kde.pdf(x_star[i]) /
                 norm.pdf(x_star[i], loc=1.1, scale=2.3)))
            
          elif dec_data[i]==1 and stream_0_1[i]==0 and x_star2[i]>0 :
            count_mc = count_mc + 1
            count = count + (
                (kde.pdf(x_star2[i]) /
                 norm.pdf(x_star2[i], loc=-1.1, scale=2.3)))

                
        # if dec_data_bpsk[i] != stream_0_1[i]:
        #     count_bpsk = count_bpsk + 1

    # count = count
    # count_mc = count_mc
    ber_est.append(count_mc / num_trials)
    #ber_est_mc.append(count_mc / num_trials)
    #ber_est_bpsk.append(count_bpsk / num_trials)
    # sim_ber_est.append(ber_est)
    # sim_ber_est_mc.append(ber_est_mc)

#print(ber_t)
print(ber_coef_theo_sum * ber_theo_coef)
# print(asarray(sim_ber_est).sum(axis=0))
# print(asarray(sim_ber_est_mc).sum(axis=0))
print(ber_est)
#print(ber_est_mc)
#print(expon.pdf(pdf))
#print(np.mean(comb_at_recv))
#print(ber_est_bpsk)

plt.semilogy(snr_range_db,
             ber_coef_theo_sum * ber_theo_coef,
             label='Theoretical')
plt.semilogy(snr_range_db, ber_est, marker='o', label='Importance Sampling')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title(
    'Bit Error Rate (BER) Estimation with Importance Sampling and Weighing Ratio'
)
plt.legend()
plt.grid(True)
plt.show()

# plt.hist(
#     comb_at_recv,
#     #  norm.pdf(comb_at_recv, loc=0, scale=1),
#     #  'r-',
#     #  lw=5,
#     bins=100,
#     alpha=0.6,
#     label='p(x)')
# plt.hist(x_star, bins=100, label='sam')

# plt.legend()
# plt.grid(True)
# plt.show()


# test1 = norm.rvs(loc=-2, scale=var_x, size=num_trials)
# test2 = norm.rvs(loc=2, scale=var_x, size=num_trials)
# x = np.linspace(min(comb_at_recv) - 1, max(comb_at_recv) + 1, 1000)

# kde = gaussian_kde(comb_at_recv)
# pdf_estimated = kde(x)
#print(pdf_estimated)
# plt.hist(
#     comb_at_recv,
#     #  norm.pdf(comb_at_recv, loc=0, scale=1),
#     #  'r-',
#     #  lw=5,
#     bins=100,
#     alpha=0.6,
#     label='p(x)')
plt.plot(x, pdf_estimated, label="Estimated PDF")

# plt.hist(exps_rt, bins=100, label='sam2')
# plt.hist(pdf, bins=100, label='sam3')

plt.legend()
plt.grid(True)
plt.show()




# (
#                 (norm.pdf(x_star[i], loc=mean_x, scale=var_x) /
#                 norm.pdf(x_star[i], loc=mean_x_star, scale=var_x_star))  *
#                 (norm.pdf(x_star2[i], loc=mean_x, scale=var_x) /
#                 norm.pdf(x_star2[i], loc=mean_x_star, scale=var_x_star)) * (norm.pdf(x_star3[i], loc=mean_x, scale=var_x) /
#                 norm.pdf(x_star3[i], loc=mean_x_star, scale=var_x_star))  
#                 #* (norm.pdf(x_star4[i], loc=mean_x, scale=var_x) /
#                 # norm.pdf(x_star4[i], loc=mean_x_star, scale=var_x_star))
#                 )