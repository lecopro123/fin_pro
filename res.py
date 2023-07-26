import random as rd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

n1 = 400
count = 0
x = uniform.rvs(0, 2, size=n1)
x_star = uniform.rvs(0.7, 1.8, size=n1)
c1 = []
x1 = []
for i in range(1, n1):
    if (x_star[i] > 0.7):
        count = count + (
            (uniform.pdf(x_star[i], 0, 2) / uniform.pdf(x_star[i], 0.7, 1.8)))
    c1.append(count / i)
    x1.append(i)

print(count / n1)

n = 650
count2 = 0
x = uniform.rvs(0, 2, size=n)
x_star = uniform.rvs(0.7, 1.8, size=n)
c = []
x11 = []
for i in range(1, n):
    if (x[i] > 0.7):
        count2 = count2 + 1
    c.append(count2 / i)
    x11.append(i)

print(count2 / n)

print(np.sqrt(((count2 / n - count / n1)**2) / 2))

true = []
st = []
for j in range(1500):
    true.append(0.65)
    st.append(j)

# plt.figure()
# plt.grid(True)
# plt.plot(x1, c1, label='MC+IS')
# plt.plot(x11, c, label='MC')
# plt.plot(st, true, label='True value')
# plt.legend()
# plt.show()

# for k in range(len(x_star)):
#     if (x_star[k] > 1.8):
#         x_star[k] = rd.random()

vals = np.arange(0.7, 200, 1)
vals_y = np.repeat(1, 200)

# l1 = np.repeat(0, 100)
# l1x = np.arange(start=0, stop=0.5, step=0.005)

# b1 = np.repeat(2, 100)
# b1x = np.arange(start=0, stop=0.5, step=0.005)

# l2 = np.repeat(0.7, 100)
# l2x = np.arange(start=0, stop=0.555, step=0.00555)

# b2 = np.repeat(2.5, 100)
# b2x = np.arange(start=0, stop=0.555, step=0.00555)

# plt.figure()
# plt.grid(True)
# plt.plot(x, uniform.pdf(x, 0, 2), color='blue', label='P(x)')
# plt.plot(l1, l1x, color='blue')
# plt.plot(b1, b1x, color='blue')
# plt.plot(x_star, uniform.pdf(x_star, 0.7, 1.8), color="black", label="Q(x)")
# plt.plot(l2, l2x, color='black')
# plt.plot(b2, b2x, color='black')
# plt.legend()
# plt.show()

plt.figure()
plt.grid(True)
plt.plot(vals, vals_y, color='blue', label='f(x)')
plt.legend()
plt.show()