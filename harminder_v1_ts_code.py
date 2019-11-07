#%%
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
from datetime import date, timedelta
import datetime

#%%
# read in data
data = pd.read_csv("//Users//harminder//Documents//test//drive-download-20191016T091937Z-001//DemandForecasting_Challenge_Data.csv")
#%%

end = datetime.datetime.strptime('17032019', "%d%m%Y").date() #end of w12 2019
start = datetime.datetime.strptime('01012017', "%d%m%Y").date() # start of week1 2017
date_index = pd.DataFrame(pd.date_range(start, end, freq='W')) # generate date range with interval 7
data.reset_index(drop=True, inplace=True)
date_index.reset_index(drop=True, inplace=True)
data = pd.concat([date_index,data],axis=1)
data.columns.values[0] = "date_ind"

#%%
data1=data
data1.set_index('date_ind',inplace=True)
data1.drop(['iso_week'],inplace=True,axis=1)
#%%
orig_data = data1
train = data1.iloc[:101]
validation = data1.iloc[101:104]
test = data1.iloc[104:]
#%%

data.plot()
plt.show()

#%%
import seaborn as sns
import matplotlib.ticker as ticker
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})

cols_plot = ['boxes', 'active1', 'active2','active3','active4']
axes = data[cols_plot].plot(marker='.', alpha=0.5, linestyle='-', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Count')
#    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
#    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

## visual: we see a pattern every 4 weeks: clear in active 3 & 4


#%%

## 
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(12).mean()
    rolstd = pd.Series(timeseries).rolling(12).std()

    fig = plt.figure(figsize=(12, 10), dpi=80)
    ax = fig.add_subplot(111)
    #Plot rolling statistics:
    ax.xaxis_date()

    orig = ax.plot(timeseries, color='blue',label='Original')
    mean = ax.plot(rolmean, color='red', label='Rolling Mean')
    std = ax.plot(rolstd, color='black', label = 'Rolling Std')
    
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, len(timeseries), 10)
    minor_ticks = np.arange(0, len(timeseries), 5)

    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


#%%
test_stationarity(train['boxes'])
# reject null hyp; stationary data
#p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
#p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.

#%%
ts_log_diff = train['boxes']
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

## based on plot start with ar 2 model

#In this plot, the two dotted lines on either sides of 0 are the confidence interevals. These can be used to determine the ‘p’ and ‘q’ values as:
#
#p – The lag value where the PACF chart crosses the upper confidence interval for the first time. If you notice closely, in this case p=2.
#q – The lag value where the ACF chart crosses the upper confidence interval for the first time. If you notice closely, in this case q=9 (ignore, we will not take  very high order)


#%%

def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def rmse(y_true, y_pred ):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log_diff, order=(2, 0, 0))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('COUNT OF BOXES; MODEL FIT order(2,0,0); RMSE: %.4f'% rmse(results_ARIMA.fittedvalues,ts_log_diff))
forecast = np.round(results_ARIMA.forecast(steps=len(test))[0],0)

#%%

from statsmodels.tsa.arima_model import ARIMA
import itertools
# using exogenous variables 
exog_var = [['active1','active2','active3','active4'],
['active1'],['active2'],['active3'],['active4'],
['active1', 'active2'],
['active1', 'active3'],
['active1', 'active4'],
['active2', 'active3'],
['active2', 'active4'],
['active3', 'active4'],
['active1', 'active2', 'active3'],
['active1', 'active2', 'active4'],
['active1', 'active3', 'active4'],
['active2', 'active3', 'active4']
]
#%%

# based 
import warnings
warnings.filterwarnings("ignore") # specify to ignore warning messages
import statsmodels.api as sm
df = pd.DataFrame([])

for i in exog_var:
    model3 = sm.tsa.statespace.SARIMAX(ts_log_diff,exog= train[i] , order=(2,0,0))  
    results_ARIMA3 = model3.fit(disp=-1)  
#    plt.plot(ts_log_diff)
#    plt.plot(results_ARIMA2.fittedvalues, color='red')
#    plt.title('RSS: %.4f'% sum((results_ARIMA3.fittedvalues-ts_log_diff)**2))
    error_perc3= mape(ts_log_diff,results_ARIMA3.fittedvalues)
    error_rmse3 = rmse(ts_log_diff,results_ARIMA3.fittedvalues)
    variables_exo = '-'.join(i)
    data = pd.DataFrame({'variables':[variables_exo], 'MAPE':[error_perc3],'rmse':[error_rmse3],'AIC':[results_ARIMA3.aic]})
    df = df.append(data)

df = df.sort_values(by=['AIC'])

#%%
df1= pd.DataFrame([])

# Define the p and q parameters to take any value between 0 and 2
p = q = range(3)
d=[0] #while the test shows no differencing needed, visually made sense for 1st order difference
# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
season_pdq = [(x[0], x[1], x[2], 4) for x in list(itertools.product(p, d, q))]

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in season_pdq:
        model3 = sm.tsa.statespace.SARIMAX(ts_log_diff,exog= train[['active1','active2','active4']] , order=param,seasonal_order=param_seasonal)  
        results_ARIMA3 = model3.fit(disp=-1)  
    #    plt.plot(ts_log_diff)
    #    plt.plot(results_ARIMA2.fittedvalues, color='red')
    #    plt.title('RSS: %.4f'% sum((results_ARIMA3.fittedvalues-ts_log_diff)**2))
        error_perc3= mape(ts_log_diff,results_ARIMA3.fittedvalues)
        error_rmse3 = rmse(ts_log_diff,results_ARIMA3.fittedvalues)
        data1 = pd.DataFrame({'param':[param],'seasonal_param':[param_seasonal],'variables':[['active1','active2','active4']], 'MAPE':[error_perc3],'rmse':[error_rmse3],'AIC':[results_ARIMA3.aic]})
        df1 = df1.append(data1)


#%%
df1 = df1.sort_values(by=['AIC'])
    
#%%
#model fit based on above iterations, with lowest aic ['active1','active2','active4'] (1,0,1) (0,0,1,4)

model_validate = sm.tsa.statespace.SARIMAX(ts_log_diff,exog= train[['active1','active2','active4']] , order=(1,0,1),seasonal_order=(0,0,1,4))  
results_validate = model_validate.fit(disp=-1)  

#%%
predict_dy = results_validate.get_prediction(start= validation.index.values.min() , end= validation.index.values.max(), exog= validation[['active1','active2','active3']])
print('Forecast:')
print(predict_dy.predicted_mean)
print('Confidence intervals:')
print(predict_dy.conf_int()) #forecast int 95%
prdn_mean = predict_dy.predicted_mean.reset_index(drop=True)
prdn_ci = predict_dy.conf_int().reset_index(drop=True)
actuals = validation['boxes']
actuals = actuals.reset_index(drop=True)
data_op = pd.concat([prdn_mean,prdn_ci,actuals],axis=1)
data_op.columns = ['pred_mean','lower_int','upper_int','actuals']
#mape
round(mape(validation['boxes'],predict_dy.predicted_mean),2) #around
round(rmse(validation['boxes'],predict_dy.predicted_mean),2) #around

#%%
train = orig_data.iloc[:104]
test = orig_data.iloc[104:]
model_validate = sm.tsa.statespace.SARIMAX(train['boxes'],exog= train[['active1','active2','active4']] , order=(1,0,1),seasonal_order=(0,0,1,4))  
results_validate = model_validate.fit(disp=-1)  
predict_dy = results_validate.get_prediction(start= test.index.values.min() , end= test.index.values.max(), exog= test[['active1','active2','active4']])
print('Forecast:')
print(predict_dy.predicted_mean)
print('Confidence intervals:')
print(predict_dy.conf_int()) #forecast int 95%

#%%
results_validate.plot_diagnostics(figsize=(7,5))
plt.show()
#%%
prdn_mean = predict_dy.predicted_mean.reset_index(drop=True)
prdn_ci = predict_dy.conf_int().reset_index(drop=True)
data_op = pd.concat([prdn_mean,prdn_ci],axis=1)
data_op.columns = ['pred_mean','lower_int','upper_int']
# %%
data_op.to_clipboard(sep=',', index=False)


# %%

# plot

ax = train['boxes'].plot(label='observed', figsize=(20, 15))
prdn_mean = predict_dy.predicted_mean
prdn_mean.plot(ax=ax, label='Forecast')
prdn_ci = predict_dy.conf_int()
ax.fill_between(prdn_ci.index,
                prdn_ci.iloc[:, 0],
                prdn_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('#Boxes')

plt.legend()
plt.show()


# %%
