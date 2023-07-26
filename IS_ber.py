import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.stats import norm
import seaborn as sns

####input fields#######################
no_sim = 10**4  ### number of simulation
snr_db = np.arange(start=-4, stop=13, step=2)  ### snr in db for plotting
ber_sim = np.zeros(len(snr_db))
ber_sim1 = np.zeros(len(snr_db))
M = 2  #no of points in bpsk constellation
m = np.arange(0, M)
constellation = 1 * np.cos(m / M * 2 * np.pi)
####input fields_ends#######################

####### transmitter #################
inputsym = []
inputsym = np.array(inputsym)
inputsym = np.random.randint(low=0, high=M, size=no_sim)

bell1 = np.random.randn(1, no_sim)
bell2 = -1.2 + np.random.randn(1, no_sim)
################################################################
s = constellation[np.array(inputsym)]  ##### constellation
gamma = []
for j, each in enumerate(snr_db):
    gamma.append(10**(each / 10))
P = sum(abs(s)**2) / (no_sim)
N0 = np.array(P / gamma)

# fig, ax1 = plt.subplots(nrows=1, ncols=1)
# ax1.plot(np.real(constellation), np.imag(constellation), '*')
# noise1 = np.sqrt(N0 / 2) * bell1[0]
# print(noise1)
for i in range(0, len(snr_db)):
    noise1 = bell1[0] * np.sqrt(N0[i] / 2)
    r = s + noise1
    #print(r)
    ####### Receiver ################
    detected_sym = (r <= 0).astype(
        int
    )  ###threshold detection   Here detected system is the function g(r) with r being RV
    ber_sim[i] = np.sum(detected_sym != inputsym) / (no_sim)

# for i in range(0, len(snr_db)):
#     noise2 = np.sqrt(N0[i] / 2) * bell2[0]
#     R = s + noise2
#     ####### Receiver ################
#     detected_sym1 = (R <= 0).astype(
#         int
#     )  ###threshold detection   Here detected system is the function g(R) with R being RV
#     ber_sim1[i] = np.sum(detected_sym1 != inputsym) / (no_sim)

for i in range(0, len(snr_db)):
    noise2 = np.sqrt(N0[i] / 2) * bell2[0]
    R = s + noise2
    ####### Receiver ################
    detected_sym1 = (R <= 0).astype(
        int
    )  ###threshold detection   Here detected system is the function g(R) with R being RV
    #print(bell2)
    for k, j in enumerate(bell2[0]):
        #print(detected_sym1[k], inputsym[k])
        if (detected_sym1[k] != inputsym[k]):
            ber_sim1[i] = ber_sim1[i] + (norm.pdf(bell1[0][k], 0, 1) /
                                         norm.pdf(bell2[0][k], -1.2, 1))
            #print(ber_sim1[i])
    ber_sim1[i] = ber_sim1[i] / no_sim

print(ber_sim, ber_sim1)
ber_theory = 0.5 * erfc(np.sqrt(10**(snr_db / 10)))
plt.figure()
plt.grid(True)
fig, ax = plt.subplots(nrows=1)
ax.semilogy(snr_db,
            ber_sim,
            color='r',
            marker='o',
            linestyle='',
            label='BPSK Sim')
ax.semilogy(snr_db,
            ber_sim1,
            color='g',
            marker='o',
            linestyle='',
            label='BPSK Sim')
ax.semilogy(snr_db, ber_theory, marker='', linestyle='-', label='BPSK Theory')
ax.set_xlabel('$E_b/N_0(dB)$')
ax.set_ylabel('BER ($P_b$)')
ax.set_title('Probability of Bit Error for BPSK over AWGN channel')
ax.set_xlim(-5, 13)
ax.grid(True)
ax.legend()
plt.show()

plt.figure()
plt.grid(True)
sns.displot([R for _ in range(10000)], label="distribution $p(x)$")
plt.show()