# Importing the necessary modules
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, uniform,norm
from autograd import jacobian

plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize'] = 14, 6

n1=10000
th=3.5
mean_x=0
var_x=1
var_x_star = 1  
mean_x_star = 3.5
x = norm.rvs(loc=mean_x, scale=var_x, size=n1)
x_star = norm.rvs(loc=mean_x_star, scale=var_x_star, size=n1)

# Initializing the random seed
random_seed = n1


cov_val = 0  #[-0.8, 0, 0.8]

mean = np.array([0, 0])




cov = np.array([[1, 0], [0, 1]])



distr = multivariate_normal(cov=cov, mean=mean, seed=random_seed)


data = distr.rvs(size=n1)

a = np.log(1 + np.absolute(data))

b = np.linalg.norm(np.log(1 + np.abs(data)), ord=np.inf)
exp = a / b
#print((5 / 1.8)**exp)
new_rv = data * (3.5 / 1.848)**exp  #4 and 2.1(1k)   2.5 and 1.2(1k)  3.5and 1.88(1k)
# new_rv_1 = data * (10 / 1.8)**exp


def matrix_function(X):
    return np.sin(X[0]) + np.cos(X[1])


jacobian_matrix_function = jacobian(matrix_function)
Jacobian = jacobian_matrix_function(data)

wei = distr.pdf(new_rv) / distr.pdf(data)
weight = np.matmul(wei, Jacobian)

#print(weight)
c1=[]
x1=[]
count = np.array([[0,0],[0,0]])
for i in range(1, n1):
    for j in range(0, 2):
        if new_rv[i][j] >= 3.5:
            count = count + weight * new_rv[i][j]
    a=count/i
    #print(a)
    #print(np.linalg.det(a))
    c1.append(np.linalg.det(a))
    x1.append(i)
    #print(count/i)

k = count / n1
print(np. linalg. det(k))

count_mc= 0
c_m = []
x_m = []
for i in range(1, n1):  #1 for line 25
    if (x[i] > th):
        #print(x[i])
        count_mc = count_mc + 1
    c_m.append(count_mc / i)
    x_m.append(i)
print("MC", count_mc / n1)

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

# Plotting the generated samples
plt.figure()
plt.plot(data[:, 0],
         data[:, 1],
         'o',
         c='lime',
         markeredgewidth=0.5,
         markeredgecolor='black')
# plt.plot(data2[:, 0],
#          data2[:, 1],
#          'o',
#          c='red',
#          markeredgewidth=0.5,
#          markeredgecolor='blue')
plt.plot(new_rv[:, 0],
         new_rv[:, 1],
         'o',
         c='purple',
         alpha=0.2,
         markeredgewidth=0.5,
         markeredgecolor='blue')
# plt.plot(new_rv_1[:, 0],
#          new_rv_1[:, 1],
#          'o',
#          c='yellow',
#          alpha=0.2,
#          markeredgewidth=0.5,
#          markeredgecolor='red')
plt.title(f'Green samples are X purple samples are Z')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')

plt.show()


true = []
st = []
for j in range(n1):
    true.append(0.000235)
    st.append(j)

plt.figure()
plt.grid()
plt.plot(x1,c1,label="IS new app")
plt.plot(x_is,c_is,label="IS app")
plt.plot(x_m,c_m,label="MC app")
plt.plot(st, true, label='True value')
plt.xlabel('Number of iterations')
plt.legend()
plt.show()
