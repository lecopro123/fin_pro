# import numpy as np

# # OFDM parameters
# num_subcarriers = 64
# num_symbols = 10
# cyclic_prefix_len = 16

# # Generate random data symbols
# data = np.random.randint(0, 2, (num_subcarriers, num_symbols))

# # IFFT
# time_domain = np.fft.ifft(data, axis=0)
# # Add cyclic prefix
# time_domain_cp = np.concatenate(
#     (time_domain[-cyclic_prefix_len:, :], time_domain), axis=0)

# # Transmitter processing complete, transmit the signal over the channel

# # Receiver processing begins

# # Remove cyclic prefix
# time_domain_no_cp = time_domain_cp[cyclic_prefix_len:, :]

# # FFT
# received_data = np.fft.fft(time_domain_no_cp, axis=0)

# # Decoding and further processing
# decoded_data = np.round(np.real(received_data)).astype(int)

# # Print original data and decoded data
# print("Original Data:")
# print(data)
# print("\nDecoded Data:")
# print(decoded_data)

import numpy as np

th = 2 * 10**(-4)
itr = 1000
pos_all = []
for l in range(1, 6):
    pos = 0  ####probability that the packet is received successfully at receiver
    for i in range(itr):
        capture = []
        flag = 0
        for j in range(l):  ####no of HARQ tx
            # OFDM parameters
            num_subcarriers = 20000
            num_symbols = 2
            cyclic_prefix_len = 16

            # Generate random data symbols
            pre_data = np.random.randint(0, 2, (num_subcarriers, 1))

            if num_symbols > 1:
                data = np.concatenate((pre_data, pre_data), axis=1)
                #print("hell0")
            else:
                #print("l0")
                data = pre_data

            # IFFT
            time_domain = np.fft.ifft(data, axis=0)

            # Add cyclic prefix
            time_domain_cp = np.concatenate(
                (time_domain[-cyclic_prefix_len:, :], time_domain), axis=0)

            # Simulating channel noise
            SNR_dB = 60.4  # Signal-to-Noise Ratio in dB
            SNR_linear = 10**(SNR_dB / 10.0
                              )  # Converting SNR from dB to linear scale
            noise_power = 1.0 / SNR_linear  # Power of the noise
            noise = np.sqrt(noise_power) * (
                np.random.randn(*time_domain_cp.shape) +
                1j * np.random.randn(*time_domain_cp.shape))

            # Add noise to the time-domain signal
            received_signal = time_domain_cp + noise

            # Receiver processing begins

            # Remove cyclic prefix
            time_domain_no_cp = received_signal[cyclic_prefix_len:, :]

            # FFT
            received_data = np.fft.fft(time_domain_no_cp, axis=0)

            # Decoding and further processing
            decoded_data = (np.round(np.real(received_data)) > 0.5).astype(int)

            count_all = []
            for j in range(num_symbols):
                count = 0
                for i in range(num_subcarriers):
                    if (data[i][j] != decoded_data[i][j]):
                        count = count + 1
                count_all.append(count / num_subcarriers)

            capture.append(count_all)

        for j in capture:
            for a in j:
                if a <= th and not flag:
                    pos = pos + 1
                    flag = 1

    pos_all.append(pos / itr)
# Print original data and decoded data
# print("Original Data:")
# print(data)
# print("\nSer:")
print(pos_all)