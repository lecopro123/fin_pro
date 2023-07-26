import random as rd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, expon, rv_continuous
from scipy import stats

ll = 0.7
ul = 2.136  #### 2 2.136 2.5

# def custom_distribution(size):
#     # pdf = 0.1
#     # if (x >= ll and x <= 1.9):
#     #     pdf = 0.1
#     #     print(x)
#     #     #cc = cc + 1
#     # elif (x >= 1.9 and x <= 2.1363636):
#     #     pdf = 0.88
#     #     print(x)
#     #     #cc1 = cc1 + 1
#     #print(cc, cc1)
#     # Return the PDF values
#     # ranges = [(0.7, 1.9), (1.91, 2.1363636)]
#     # probabilities = [0.1, 0.88]

#     # # Initialize the probability density function (PDF) array
#     # pdf = np.zeros_like(x)

#     # # Compute the PDF based on the ranges and probabilities
#     # for range_, prob in zip(ranges, probabilities):
#     #     mask = np.logical_and(x >= range_[0], x <= range_[1])
#     #     pdf[mask] = prob

#     # # Normalize the PDF
#     # pdf /= np.sum(pdf)
#     #
#     random_numbers = np.zeros(size)

#     # Generate random numbers in the desired ranges with respective probabilities
#     for i in range(size):
#         # Generate a random number between 0 and 1
#         rand = np.random.random()

#         # Assign a value based on the random number and the desired probabilities
#         if 0.7 <= rand < 1.9:
#             random_numbers[i] = np.random.choice([0.1])
#         elif 1.9 <= rand <= 2.1363636:
#             random_numbers[i] = np.random.choice([0.88])

#     return random_numbers

# #my_cv = my_pdf()
# custom_dist = stats.rv_continuous()
# custom_dist._pdf = custom_distribution
# # formula = lambda x: x >= ll and x <= 1.9 (x=0.1)

n1 = 10000
count = 0
x = uniform.rvs(0, 2, size=n1)
x_star = uniform.rvs(1.9, 0.1, size=n1)
c1 = []
x1 = []
# k = [0.1, 0.8, 0.9]

# plt.figure()
# plt.grid(True)
# plt.plot(k, uniform.pdf(k, 0.1, 0.9), color="black", label="Q(x*)")
# plt.legend()
# plt.show()

for i in range(1, n1):
    if (x_star[i] >= 1.9):
        count = count + (
            (uniform.pdf(x_star[i], 0, 2) / uniform.pdf(x_star[i], 1.9, 0.1)))
    c1.append(count / i)
    x1.append(i)
    #if (uniform.pdf(x_star[i], 1.9, 0.1) == 0): print("inf cond")

print(count / n1)

count2 = 0
c = []
x11 = []
for i in range(1, n1):
    if (x[i] >= 1.9):
        count2 = count2 + 1
    c.append(count2 / i)
    x11.append(i)

print(count2 / n1)

#print(np.sqrt(((count2 / n1 - count / n1)**2) / 2))

true = []
st = []
for j in range(n1):
    true.append(0.05)
    st.append(j)
#print(c1)
plt.figure()
plt.grid(True)
plt.plot(x1, c1, label='MC+IS', color="red")
plt.plot(x11, c, label='MC')
plt.plot(st, true, label='True value', alpha=0.4)
#plt.xaxis.zoom(1000, 2000)
plt.legend()
plt.show()

vals = np.arange(1.9, 20, 0.1)
vals_y = np.repeat(1, 181)

l1 = np.repeat(0, 100)
l1x = np.arange(start=0, stop=0.5, step=0.005)

b1 = np.repeat(2, 100)
b1x = np.arange(start=0, stop=0.5, step=0.005)

# l11 = np.repeat(1.9, 100)
# l11x = np.arange(start=0, stop=0.5, step=0.005)

# b11 = np.repeat(2, 100)
# b11x = np.arange(start=0, stop=0.5, step=0.005)

l2 = np.repeat(1.9, 100)
l2x = np.arange(start=0, stop=10, step=0.1)

b2 = np.repeat(2, 100)
b2x = np.arange(start=0, stop=10, step=0.1)

# xs = []
# for j in x_star:
#     if (j >= ll and j <= ul):
#         xs.append(j)

plt.figure()
plt.grid(True)
plt.plot(x, uniform.pdf(x, 0, 2), color='blue', label='P(x)')
plt.plot(l1, l1x, color='blue')
plt.plot(b1, b1x, color='blue')
plt.plot(x_star,
         uniform.pdf(x_star, 1.9, 0.1),
         lw=1,
         color="black",
         label="Q(x*)")
#plt.hist(x_star, 100)
plt.plot(l2, l2x, color='black')
plt.plot(b2, b2x, color='black')
plt.legend()
plt.show()

l3 = np.repeat(1.9, 100)
l3x = np.arange(start=0, stop=1, step=0.01)
plt.figure()
plt.grid(True)
plt.plot(vals, vals_y, color='red', label='f(x)')
plt.plot(l3, l3x, color='red')
plt.legend()
plt.show()

# x1 = []
# for j in x:
#     if (j >= ll and j <= 2):
#         x1.append(j)
# plt.figure()
# plt.grid(True)
# plt.plot(x1, uniform.pdf(x1, 0.7, 1.3), color='yellow', label='P(x)F(x)')
# plt.plot(x_star,
#          uniform.pdf(x_star, 1.9,0.1),
#          lw=1,
#          alpha=0.6,
#          color="r",
#          label="Q(x*)")
# plt.plot(l11, l11x, color='yellow')
# plt.plot(b11, b11x, color='yellow')
# plt.legend()
# plt.show()