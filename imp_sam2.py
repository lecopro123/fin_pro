import random as rd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

n1 = 6000
count = 0
x = uniform.rvs(0, 2, size=n1)
x_star = uniform.rvs(0.7, 2.5, size=n1)
c1 = []
x1 = []

for i in range(1, n1):
    if (x_star[i] > 0.7):  ### this will work with th=0.7
        count = count + (
            (uniform.pdf(x_star[i], 0, 2) / uniform.pdf(x_star[i], 0.7, 2.5)))
    c1.append(count / i)
    x1.append(i)
    if (uniform.pdf(x_star[i], 0.7, 2.5) == 0): print("inf cond")

print(count / n1)

count2 = 0
c = []
x11 = []
for i in range(1, n1):
    if (x[i] >= 0.7
        ):  #### this wont work with th=0.7 you have to precise with the limit
        count2 = count2 + 1
    c.append(count2 / i)
    x11.append(i)

print(count2 / n1)

#print(np.sqrt(((count2 / n1 - count / n1)**2) / 2))

true = []
st = []
for j in range(6000):
    true.append(0.65)
    st.append(j)

plt.figure()
plt.grid(True)
plt.plot(x1, c1, label='MC+IS')
plt.plot(x11, c, label='MC')
plt.plot(st, true, label='True value')
#plt.xaxis.zoom(1000, 2000)
plt.legend()
plt.show()

vals = np.arange(0.7, 20, 0.1)
vals_y = np.repeat(1, 193)

l1 = np.repeat(0, 100)
l1x = np.arange(start=0, stop=0.5, step=0.005)

b1 = np.repeat(2, 100)
b1x = np.arange(start=0, stop=0.5, step=0.005)

l2 = np.repeat(0.7, 100)
l2x = np.arange(start=0, stop=0.555, step=0.00555)

b2 = np.repeat(2.5, 100)
b2x = np.arange(start=0, stop=0.555, step=0.00555)

plt.figure()
plt.grid(True)
plt.plot(x, uniform.pdf(x, 0, 2), color='blue', label='P(x)')
plt.plot(l1, l1x, color='blue')
plt.plot(b1, b1x, color='blue')
plt.plot(x_star, uniform.pdf(x_star, 0.7, 1.8), color="black", label="Q(x*)")
#plt.hist(x_star, 100)
plt.plot(l2, l2x, color='black')
plt.plot(b2, b2x, color='black')
plt.legend()
plt.show()

l3 = np.repeat(0.7, 100)
l3x = np.arange(start=0, stop=1, step=0.01)
plt.figure()
plt.grid(True)
plt.plot(vals, vals_y, color='red', label='f(x)')
plt.plot(l3, l3x, color='red')
plt.legend()
plt.show()

plt.figure()
plt.grid(True)
plt.plot(x_star, uniform.pdf(x_star, 0.7, 1.8), color="black", label="Q(x)")
plt.plot(l2, l2x, color='black')
plt.plot(b2, b2x, color='black')
plt.legend()
plt.show()
print((count2 / n1 - count / n1)**2)
