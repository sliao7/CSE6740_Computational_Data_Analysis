import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.sandbox.regression.predstd import wls_prediction_std

save_path = '../Latex/images/'

rng = default_rng(seed = 1)
m = 1000 # number of points
n = 6 # number of features/variables

x = np.ones((m,n))
error_sigma = .5
error_sigma5 = .25
error_sigma6 = .6


x[:,1] = rng.standard_normal(m)
x[:,2] = rng.standard_normal(m)
x[:,3] = .5 * x[:,1] + rng.standard_normal(m)/5
x[:,4] = np.exp(rng.standard_normal(m))
x[:,5] = x[:,1] + 2 * x[:,2] + .5 * x[:,3] + error_sigma6 * rng.standard_normal(m)
eps = error_sigma * rng.standard_normal(m)


y = 2 * x[:,0] + 2 * x[:,1] + 3 * x[:,1] **2 + x[:,2] + x[:,3] + 3 * np.log(x[:,4]) + 3 * x[:,2] * eps + 2 * eps

# add outliers
y[0] = 40
y[1] = -15

# add high leverage points
x[2,3] = 2 * x[2,1] - 3
x[3,1] = -4

# plt.plot(x[:,1], x[:,3], '.')
# plt.show()

## Scatter plot
# plt.figure(figsize = (10,8))
# for i in range(n):
#     plt.subplot(2, 3, i+1)
#     plt.scatter(x[0:2,i],y[0:2], c = 'green')
#     plt.scatter(x[2:4,i],y[2:4], c = 'red')
#     plt.plot(x[:,i], y,'.',alpha=0.1)    
#     plt.title('Scatter of x' + str(i) + ' and y')
#     plt.xlabel('x' + str(i))
#     plt.ylabel('y')
# plt.subplots_adjust(hspace=.5, wspace = .5)
# plt.savefig(save_path + 'Scatter.jpg')
# plt.close()
# plt.show()

# for i in range(1,7):
#     plt.subplot(2, 3, i)
#     plt.hist(x[:,i])
#     plt.title('Histigram of x' + str(i))
# plt.subplots_adjust(hspace=.5)
# plt.show()

x = pd.DataFrame(x)

# baseline model
est = sm.OLS(y, x)
est1 = est.fit()
# print('Baseline Model summary:')
# print(est1.summary())




# remove outliers 
ypred = est1.predict(x)
residual = y - ypred
z = np.abs(stats.zscore(residual))
print(np.where(z > 3)[0].shape)
x = x[z<3]
y = y[z<3]
# print(z[z>3])


## after remove the outliers, the two outliers y = -15 and y - 40 are removed
# 11 outliers were removed

est = sm.OLS(y, x)
est1 = est.fit()
# print('Model after removing outliers:')
# print(est1.summary())


# plot_res(x,y)
# ypred = est1.predict(x)
# residual = y - ypred
# z = np.abs(stats.zscore(residual))
# print(z[z>3])
# print(y[y == -15])

# remove high leverage points

# sm.graphics.influence_plot(est1,alpha = .3)
# plt.show()

H = x @ np.linalg.inv(x.T @ x) @ x.T 
leverage = np.diag(H)

print(np.where(leverage > 3 * n / m)[0].shape)

# remove high leverage points
x = x[leverage < 3 * n / m]
y = y[leverage < 3 * n / m]


est = sm.OLS(y, x)
est1 = est.fit()
# print('Model after removing high leverage points:')
# print(est1.summary())



## Scatter plot after removing outliers and high leverage points
# plt.figure(figsize = (10,8))
# for i in range(n):
#     plt.subplot(2, 3, i+1)
#     plt.plot(x[i], y - est1.predict(x),'.',alpha=0.4)
#     plt.title('Scatter of x' + str(i) + ' and residual')
#     plt.xlabel('x' + str(i))
#     plt.ylabel('residual')
# plt.subplots_adjust(hspace=.5, wspace = .5)
# plt.savefig(save_path + 'Scatter_cleaned.jpg')
# plt.close()
# plt.show()

## Nonlinearity of x1 and x3
x[6] = x[1]**2 
x[7] = x[3]**2

# plot nonlinear transformation of x4
# plt.subplot(1,2,1)
# plt.hist(x[4])
# plt.title('Histigram of x4')

# # plt.subplot(1,3,2)
# # plt.hist(np.exp(x[4]))

# plt.subplot(1,2,2)
# plt.hist(np.log(x[4]))
# plt.title('Histigram of log(x4)')
# plt.savefig(save_path + 'hist')
# plt.show()

x[8] = np.log(x[4])
est = sm.OLS(y, x)
est1 = est.fit()
# print('Model after adding three nonlinear terms')
# print(est1.summary())





# est = sm.OLS(y, x)
# est1 = est.fit()
# print('Model after adding two nonlinear terms:')
# print(est1.summary())
# print(est1.summary().as_latex())

# check conditional number
# print('The condition number is: ', np.linalg.cond(x))

# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x.columns 
  
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x.values, i) 
                          for i in range(len(x.columns))] 
# print('VIF:')
# print(vif_data)
# print(vif_data.to_latex(index=False))
# est = sm.OLS(y, x[[0,1,2,3,4,6]])
# est1 = est.fit()
# print(est1.summary())

x = x.drop([5],axis = 1)
vif_data = pd.DataFrame() 
vif_data["feature"] = x.columns 
vif_data["VIF"] = [variance_inflation_factor(x.values, i) 
                          for i in range(len(x.columns))] 
# print(vif_data)
# print(vif_data.to_latex(index=False))



x = x.drop([3],axis = 1)
vif_data = pd.DataFrame() 
vif_data["feature"] = x.columns 
vif_data["VIF"] = [variance_inflation_factor(x.values, i) 
                          for i in range(len(x.columns))] 
# print(vif_data)
# print(vif_data.to_latex(index=False))

est = sm.OLS(y, x)
est1 = est.fit()
# print(est1.summary())



## Scatter plot after removing high vif features
# j = 1
# for i in x.columns:
#     plt.subplot(2, 3, j)
#     plt.plot(x[i], y,'.',alpha=0.4)
#     plt.title('Scatter of x' + str(i) + ' and y')
#     j += 1
# plt.subplots_adjust(hspace=.5)
# # plt.savefig(save_path + 'Scatter_cleaned.jpg')
# # plt.close()
# plt.show()

x = x.drop([4],axis = 1)
est = sm.OLS(y, x)
est1 = est.fit()
# print('Model after dropping x4')
# print(est1.summary())
# print(est1.summary().as_latex())

vif_data = pd.DataFrame() 
vif_data["feature"] = x.columns 
vif_data["VIF"] = [variance_inflation_factor(x.values, i) 
                          for i in range(len(x.columns))] 
# print(vif_data)
# print(vif_data.to_latex(index=False))

x = x.drop([7],axis = 1)
est = sm.OLS(y, x)
est1 = est.fit()
# print('Model after dropping x4')
# print(est1.summary())
# print(est1.summary().as_latex())


# ## Plot residule
# def plot_res(x,y):
#     ypred = est1.predict(x)
#     residual = y - ypred
#     plt.plot(x[2],residual,'.',alpha=0.5)
#     # plt.savefig(save_path + 'residual.jpg')
#     # plt.close()
#     plt.show()

# plot_res(x,y)

ypred = est1.predict(x)
residual = y - ypred

RSS = sum((residual)**2)
TSS = sum((y - np.mean(y))**2)
print(1 - RSS/TSS)

## Scatter plot of residuals vs predictors
plt.figure(figsize = (10,8))
j = 1
for i in x.columns:
    plt.subplot(2, 3, j)
    plt.plot(x[i], residual,'.',alpha=0.4)
    plt.title('Scatter of x' + str(i) + ' and residual')
    plt.xlabel('x' + str(i))
    plt.ylabel('residual')
    j += 1
plt.subplot(2, 3, j)
plt.plot(ypred, residual,'.',alpha=0.4)
plt.xlabel('ypred')
plt.ylabel('residual')
plt.title('Scatter of ypred and residual')
plt.subplots_adjust(hspace=.5, wspace = .5)
plt.savefig(save_path + 'Scatter_res.jpg')
plt.close()
# plt.show()


est = sm.OLS(np.abs(residual), x[[0,2,6]])
est1 = est.fit()
print(est1.summary())
sigma_pred = est1.predict(x[[0,2,6]])
# print(sigma_pred.shape)


mod_wls = sm.WLS(y, x, weights=1./(sigma_pred ** 2))
res_wls = mod_wls.fit()
print(res_wls.summary())
print(res_wls.summary().as_latex())

w = 1./(sigma_pred ** 2)
ypred = res_wls.predict(x)
residual = y - ypred
RSS = sum(w * (residual)**2)
weighted_mean = sum(w * y)/sum(w)
TSS = sum(w * (y - weighted_mean)**2)
print(TSS, RSS)
print(1 - RSS/TSS)

plt.plot(1/sigma_pred**2,'.')
plt.show()

## Scatter plot of residuals vs predictors
ypred = res_wls.predict(x)
residual = y - ypred
plt.figure(figsize = (10,8))
j = 1
for i in x.columns:
    plt.subplot(2, 3, j)
    plt.plot(x[i], residual,'.',alpha=0.4)
    plt.title('Scatter of x' + str(i) + ' and residual')
    plt.xlabel('x' + str(i))
    plt.ylabel('residual')
    j += 1
plt.subplot(2, 3, j)
plt.plot(ypred, residual,'.',alpha=0.4)
plt.xlabel('ypred')
plt.ylabel('residual')
plt.title('Scatter of ypred and residual')
plt.subplots_adjust(hspace=.5, wspace = .5)
plt.savefig(save_path + 'Scatter_res1.jpg')
plt.close()
# plt.show()
# plt.close()

# plt.close()




# x = x.drop([2],axis = 1)
# mod_wls = sm.WLS(y, x, weights=1./(sigma_pred ** 2))
# res_wls = mod_wls.fit()
# print(res_wls.summary())


# ypred = res_wls.predict(x)
# residual = y - ypred
# est = sm.OLS(np.abs(residual), x[[0,2,6]])
# est1 = est.fit()
# print(est1.summary())
# sigma_pred = est1.predict(x[[0,2,6]])
# mod_wls = sm.WLS(y, x, weights=1./(sigma_pred ** 2))
# res_wls = mod_wls.fit()
# print(res_wls.summary())


# ypred = res_wls.predict(x)
# residual = y - ypred
# est = sm.OLS(np.abs(residual), x[[0,2,6]])
# est1 = est.fit()
# print(est1.summary())
# sigma_pred = est1.predict(x[[0,2,6]])
# mod_wls = sm.WLS(y, x, weights=1./(sigma_pred ** 2))
# res_wls = mod_wls.fit()
# print(res_wls.summary())
# # print(res_wls.summary().as_latex())

# ypred = res_wls.predict(x)
# residual = y - ypred
# est = sm.OLS(np.abs(residual), x[[0,2,6]])
# est1 = est.fit()
# print(est1.summary())
# sigma_pred = est1.predict(x[[0,2,6]])
# mod_wls = sm.WLS(y, x, weights=1./(sigma_pred ** 2))
# res_wls = mod_wls.fit()
# print(res_wls.summary())



# ## Scatter plot of residuals vs predictors
# ypred = res_wls.predict(x)
# residual = y - ypred
# plt.figure(figsize = (10,8))
# j = 1
# for i in x.columns:
#     plt.subplot(2, 3, j)
#     plt.plot(x[i], residual,'.',alpha=0.4)
#     plt.title('Scatter of x' + str(i) + ' and residual')
#     plt.xlabel('x' + str(i))
#     plt.ylabel('residual')
#     j += 1
# plt.subplot(2, 3, j)
# plt.plot(ypred, residual,'.',alpha=0.4)
# plt.xlabel('ypred')
# plt.ylabel('residual')
# plt.title('Scatter of ypred and residual')
# plt.subplots_adjust(hspace=.5, wspace = .5)
# plt.savefig(save_path + 'Scatter_res1.jpg')
# plt.close()
# # plt.show()
# w = 1./(sigma_pred ** 2)
# ypred = res_wls.predict(x)
# residual = y - ypred
# RSS = sum((w * residual)**2)
# TSS = sum((w * y - np.mean(w * y))**2)
# print(1 - RSS/TSS)

# plt.plot(1/sigma_pred**2,'.')
# plt.show()
