import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm, rayleigh, expon,gaussian_kde
from scipy import special as sp
from numpy import asarray

# Simulation parameters
# this is a 2X2 MRC system

######### input_fields ########
db_16_ber = 1.73115703 * 10**(-5)
db_14_ber = 6.38754163 * 10**(-5)
num_trials = 10  # Number of trials (sym transmitted per trial)

snr_db = 16
# ber_sim = np.zeros(len(snr_db))
M = 4 #########QPSK constellation constant
m = np.arange(1, 2*M, 2)
x1 = np.cos(m / M * np.pi)
x2 = np.sin(m / M * np.pi)
constellation = x1 + 1j * x2 #############anti-clockwise rotation

stream_0_1_b1 = np.random.randint(low=0, high=M-2, size=num_trials)
stream_0_1_b2 = np.random.randint(low=0, high=M-2, size=num_trials)

symbol_real=[]
symbol_img=[]
for j in range(0, len(stream_0_1_b1)):
    if stream_0_1_b1[j] == 0 and stream_0_1_b2[j] == 0:
        symbol_real.append(np.real(constellation[0]))
        symbol_img.append(np.imag(constellation[0]))
    elif stream_0_1_b1[j] == 0 and stream_0_1_b2[j] == 1:
        symbol_real.append(np.real(constellation[1]))
        symbol_img.append(np.imag(constellation[1]))
    elif stream_0_1_b1[j] == 1 and stream_0_1_b2[j] == 0:
        symbol_real.append(np.real(constellation[2]))
        symbol_img.append(np.imag(constellation[2]))
    elif stream_0_1_b1[j] == 1 and stream_0_1_b2[j] == 1:
        symbol_real.append(np.real(constellation[3]))
        symbol_img.append(np.imag(constellation[3]))
########################################################################
mean_x = 0
var_x = 2

noise1 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)
noise2 = norm.rvs(loc=mean_x, scale=var_x, size=num_trials)

h11 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  ####channel coefficients
h12 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  ## channel coefficients
h21 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  #### channel coefficients
h22 = norm.rvs(loc=mean_x, scale=var_x,
              size=num_trials)  ####channel coefficients

P = 1
snr = 10**(snr_db / 10)
th=1/np.sqrt(snr)
N0 = P / snr
#print(len(symbol_img),len(symbol_real))
recv_1 = (h11 * symbol_real)+(h12 * symbol_img)+ noise1 * np.sqrt(N0 / 2)
recv_2 = (h21 * symbol_real)+(h22 * symbol_img)+ noise2 * np.sqrt(N0 / 2)

############################################ZF receiver
tl=h11**2+h21**2
tr_bl=h11*h12+h21*h22
br=h12**2+h22**2
for i in range(len(tl)):
    hth=np.array([tl[i],tr_bl[i]],[tr_bl[i],br[i]])
recovered_sym_1=tl*recv_1+tr_bl*recv_2
recovered_sym_2=br*recv_1+tr_bl*recv_2
print(h)
h_trans_h = np.dot(h.transpose(), h)
#print(h_trans_h)
h_trans_h_inv = np.linalg.inv(h_trans_h)

psedo_inv = np.dot(h_trans_h_inv,h.transpose())

#theta_cap = np.dot(psedo_inv, close)