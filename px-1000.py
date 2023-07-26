import random as rd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

n1 = 50000
var_x = 1  #####300
mean_x = 0
var_x_star = 1  ###60
mean_x_star = 2.9  ####110 1015
th = 3  ###100
x = norm.rvs(loc=mean_x, scale=var_x, size=n1)
x_star = norm.rvs(loc=mean_x_star, scale=var_x_star, size=n1)

#### f(x)=1 for x>1000
#### p(X)=standard normal distribution with mean=0 varience=1 , where X is a R.V
#### g(X_star)=normal distribution with mean 1000 and varience 1, where X_star is a R.V
count = 0
c_m = []
x_m = []
for i in range(1, n1):  #1 for line 25
    if (x[i] > th):
        #print(x[i])
        count = count + 1
    c_m.append(count / i)
    x_m.append(i)
print("MC", count / n1)

count_is = 0
c_is = []
x_is = []
for j in range(1, n1):  ##1 for line 36
    if (x_star[j] > th):
        #print(x_star[j])
        count_is = count_is + (
            (norm.pdf(x_star[j], loc=mean_x, scale=var_x) /
             norm.pdf(x_star[j], loc=mean_x_star, scale=var_x_star)))
    c_is.append(count_is / j)
    x_is.append(j)
print("MC+IS", count_is / n1)

print("True value of P(X>1000) is 0.00135")

true = []
st = []
for j in range(n1):
    true.append(0.00135)
    st.append(j)

plt.figure()
plt.grid(True)
plt.plot(x_is, c_is, label='MC+IS')
plt.plot(x_m, c_m, label='MC')
plt.plot(st, true, label='True value')
#plt.xaxis.zoom(1000, 2000)
plt.legend()
plt.show()

x_f = []
for j in x:
    if j >= 3:
        x_f.append(j)

plt.figure()
plt.grid(True)
plt.plot(x,
         norm.pdf(x, loc=mean_x, scale=var_x),
         'r-',
         lw=5,
         alpha=0.6,
         label='p(x)')
# plt.plot(x_star,
#          norm.pdf(x_star, loc=mean_x_star, scale=var_x_star),
#          'b-',
#          lw=5,
#          alpha=0.6,
#          label='q(x*)')
#plt.hist(x, bins=100)
#plt.hist(x_star, bins=100)
plt.legend()
plt.show()

vals = np.arange(th, 20 + th, 0.1)
vals_y = np.repeat(1, 200)
l3 = np.repeat(th, 100)
l3x = np.arange(start=0, stop=1, step=0.01)

plt.figure()
plt.grid()
plt.plot(vals, vals_y, color='green', label='f(x)')
plt.plot(l3, l3x, color='green')
plt.legend()
plt.show()
