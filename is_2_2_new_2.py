import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, uniform, norm
from autograd import jacobian
                        ###########loss function : x+3y>=8.5
n1=10000000
mean_x=0
var_x=1
sam_mean_x=2
sam_var_x=2
th=8.5
x1 = norm.rvs(loc=mean_x, scale=var_x, size=n1)
x2 = norm.rvs(loc=mean_x, scale=var_x, size=n1)
sam_x1 = norm.rvs(loc=sam_mean_x, scale=sam_var_x, size=n1)
sam_x2 = norm.rvs(loc=sam_mean_x, scale=sam_var_x, size=n1)

###########################new app IS
a1 = np.log(1 + np.absolute(x1))
a2= np.log(1 + np.absolute(x2))
b=np.log(1.88)
exp2=a2/b
exp1=a1/b
new_x1 = x1 * (th /5.9)**exp1  
new_x2 = x2 * (th /5.27)**exp2  

# j1k=1+((np.absolute(x1)/(1 + np.absolute(x1)))*(np.log(3.5/1.88)/np.log(1.88)))
# j2k=1+((np.absolute(x2)/(1 + np.absolute(x2)))*(np.log(3.5/1.88)/np.log(1.88)))
# jac1=j1k*j2k*(3.5 /1.88)**(exp1)
# jac2=j1k*j2k*(3.5 /1.88)**(exp2)
# #jac=j1k*j2k*(3.5 /1.88)**(exp2+exp1)
# weight1=(norm.pdf(new_x1,loc=np.mean(new_x1),scale=np.var(new_x1))*jac1)/(norm.pdf(x1,loc=np.mean(x1),scale=np.var(x1)))
# weight2=(norm.pdf(new_x2,loc=np.mean(new_x2),scale=np.var(new_x2))*jac2)/(norm.pdf(x2,loc=np.mean(x2),scale=np.var(x2)))
weight1=(norm.pdf(new_x1,loc=np.mean(x1),scale=np.var(x1)))/(norm.pdf(new_x1,loc=np.mean(new_x1),scale=np.var(new_x1)))  #weight of 1st dim
weight2=(norm.pdf(new_x2,loc=np.mean(x2),scale=np.var(x2)))/(norm.pdf(new_x2,loc=np.mean(new_x2),scale=np.var(new_x2)))  #weight of 2nd dim

count1=0
#count2=0
c1 = []
x_1 = []

for i in range(1,n1):   ###looping through the samples
        if new_x1[i] + 3*new_x2[i] >= th:  ####checking acc to the 3rd step of Algo 1
                # if weight1[i]>weight2[i]:
                #     k=weight1[i]/(weight1[i]+weight2[i])
                # elif weight1[i]<weight2[i]:
                #     k=weight2[i]/(weight1[i]+weight2[i])
                # else:
                #     k=weight2[i]/(weight1[i]+weight2[i])
                count1 = count1 + weight1[i]*weight2[i]
                #count2 = count2 + weight2[i]
        c1.append(count1/i)
        x_1.append(i)
print("IS new app",count1/n1)
#print(count2/(n1))

###########################vanilla MC
count_mc=0
c_m = []
x_m = []
#print(len(x1))
for i in range(1,n1):      ###looping through the samples
        #print(x1[i],i)
        if x1[i] + 3*x2[i] >= th:    ####checking acc to the MC approach
                count_mc = count_mc+1
        c_m.append(count_mc / i)
        x_m.append(i)
print("MC",count_mc / n1)


##########################MC+IS
count_is = 0
c_is = []
x_is = []
#wei = distr.pdf(sam) / distr_sam.pdf(sam)
#print(wei)
for i in range(1,n1):  #looping through the samples
        if (sam_x1[i] + 3*sam_x2[i] >= th):
                k=(
            (norm.pdf(sam_x1[i], loc=mean_x, scale=var_x) /
             norm.pdf(sam_x1[i], loc=sam_mean_x, scale=sam_var_x))* (norm.pdf(sam_x2[i], loc=mean_x, scale=var_x) /
             norm.pdf(sam_x2[i], loc=sam_mean_x, scale=sam_var_x)))
                count_is = count_is +  k
                #print(k)
        c_is.append(count_is / i)
        x_is.append(i)
print("MC+IS", count_is / n1)



true = []
st = []
for j in range(n1):
    true.append(0.00359)
    st.append(j)
plt.figure()
plt.plot(x1,x2,
         'o',
         c='lime',
         markeredgewidth=0.5,
         markeredgecolor='black')
plt.plot(new_x1,new_x2,
         'o',
         c='red',
         alpha=0.2,
         markeredgewidth=0.5,
         markeredgecolor='black')
#plt.title(f'Green samples are X purple samples are Z')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')
plt.show()

plt.figure()
plt.grid()
plt.plot(x_1, c1, label="IS new app")
plt.plot(x_is, c_is, label="IS app")
plt.plot(x_m, c_m, label="MC app")
plt.plot(st, true, label='True value')
plt.xlabel('Number of iterations')
plt.legend()
plt.show()