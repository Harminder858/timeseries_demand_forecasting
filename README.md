# timeseries_demand_forecasting

1. Exploratory:
  FOR ANALYSIS AND FORECASTING, DATA IS SPLIT INTO 3:train, test and validation
  (train = data1.iloc[:101]; validation = data1.iloc[101:104]; test = data1.iloc[104:])


![alt text](https://github.com/Harminder858/timeseries_demand_forecasting/blob/master/Picture%201.png)


Initial visual cue point towards a pattern (seasonality?) every for 4 weeks. More apparent in ’active3’ and ‘active4’ variables
Also, there is no clear trend(upwards or downwards), that is apparent  from the plot
 

 2. Checking FOR STATIONARITY:
 
To diagnose stationarity, we look at:
1.Rolling Statistics: Moving average and St.Dev/Variance plots
-  Rolling mean is constant over time
Rolling std. is constant over time
Inference: Stationary Data
2.Dickey Fuller test for stationarity
p-value <= 0.05 
Enough evidence to reject the null hypothesis (H0), the data does not have a unit root and is stationary

![alt text](https://github.com/Harminder858/timeseries_demand_forecasting/blob/master/picture2.png)

Results of Dickey-Fuller Test: 
Test Statistic -4.634444 
p-value 0.000112
#Lags Used 1.000000 
Number of Observations Used 99.000000 
Critical Value (1%) -3.498198 
Critical Value (5%) -2.891208 
Critical Value (10%) -2.582596 

3. ACF & PACF PLOTS

![alt text](https://github.com/Harminder858/timeseries_demand_forecasting/blob/master/Picture3.png)

#In this plot, the two dotted lines on either sides of 0 are the confidence interevals. These can be used to determine the ‘p’ and ‘q’ values as:
#p: The lag value where the PACF chart crosses the upper confidence interval for the first time. If you notice closely, in this case p=2.
#q: The lag value where the ACF chart crosses the upper confidence interval for the first time. If you notice closely, in this case q=9                  (ignored, we will not take very high order to avoid overfit)


4. BASELINE MODEL
We establish a baseline model, and look to improve iterations based on RMSE and AIC

![alt text](https://github.com/Harminder858/timeseries_demand_forecasting/blob/master/picture4.png)

5. ARIMAX ITERATIONS
 
 INITIALLY NARROWED DOWN ON WHAT COMBINATION OF EXOG VARIABLES TO USE AS REGRESSORS IN ARIMA:  ['active1','active2','active3','active4’]

![alt text](https://github.com/Harminder858/timeseries_demand_forecasting/blob/master/Picture_Table.png)

6. NEXT WE ITERATE OVER (P,d,Q)& SEASONAL (P,d,Q)  VALUES b/W (0,2) for p & q
   (for lowest AIC)

![alt text](https://github.com/Harminder858/timeseries_demand_forecasting/blob/master/Picture5.png)

7. VALIDATION SET

MODEL 1:                                                                     MODEL 2:
[active1,active2,active4],                                                  [active1,active2,active3],

PDQ AND SEASONAL PDQ: [1,0,1], [0,0,1,4]                                    PDQ AND SEASONAL PDQ: [1,0,1], [0,0,1,4]

 
![alt text](https://github.com/Harminder858/timeseries_demand_forecasting/blob/master/Picture6.png)

MAPE: 1.57 %                                                                 MAPE: 7.81 %
RMSE: 613.65                                                                 RMSE: 1821.13	

forecast model parameters to be considered from model 1 [active1,active2,active4],PDQ AND SEASONAL PDQ: [1,0,1], [0,0,1,4]

8. RESIDUAL DIAGNOSTICS

![alt text](https://github.com/Harminder858/timeseries_demand_forecasting/blob/master/Picture7.png)

- Top right plot, red KDE line follows closely with the N(0,1) line (where N(0,1)) is the standard notation for a normal distribution with mean 0 and standard deviation of 1), indicating that the residuals are normally distributed.
- The qq-plot on the bottom left shows that the ordered distribution of residuals (blue dots) follows the linear trend of the samples taken from a standard normal distribution with N(0,1), indicating that the residuals are normally distributed.
- The residuals over time (top left plot) don’t display any obvious seasonality and appear to be white noise. 
- This is confirmed by the autocorrelation (i.e. correlogram) plot on the bottom right, which shows that the time series residuals have low correlation with lagged versions of itself.

9. 12 WEEK FORECAST

![alt text](https://github.com/Harminder858/timeseries_demand_forecasting/blob/master/Picture8.png)
 
![alt text](https://github.com/Harminder858/timeseries_demand_forecasting/blob/master/forecast_values.png)
 

