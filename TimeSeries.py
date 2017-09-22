
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:




# In[2]:

train = pd.read_csv("~/Documents/Nitin/TestFiles/AirPassengers.csv")


# In[ ]:




# In[3]:

get_ipython().magic('matplotlib inline')


# In[ ]:




# In[4]:

from matplotlib.pylab import rcParams


# In[ ]:




# In[5]:

rcParams['figure.figsize'] = 15, 6


# In[ ]:




# In[7]:

train.head()


# In[ ]:




# In[12]:

train.dtypes


# In[6]:

dateparse = lambda dates:pd.datetime.strptime(dates,'%Y-%m')


# In[7]:

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')


# In[8]:

train['Month'] = train['Month'].apply(dateparse)


# In[10]:

train.head()


# In[24]:

train.index


# In[14]:

data = train.set_index(train['Month'])


# In[ ]:




# In[ ]:




# In[ ]:




# In[12]:

data.head()


# In[ ]:




# In[27]:

data.index


# In[15]:

ts = data['#Passengers']
ts.head()


# In[ ]:




# In[ ]:




# In[ ]:




# In[12]:


from statsmodels.tsa.stattools import adfuller


# In[ ]:




# In[11]:

def testStationary(timeseries):
    rolmean = pd.rolling_mean(timeseries, window =12)
    #rolmean = pd.rolling(timeseries,window=12,center=False).mean()
    rolstd = pd.rolling_std(timeseries, window = 12)
    #rolstd = pd.rolling(timeseries,window=12,center=False).std()
    orig = plt.plot(timeseries, color = 'blue',label = 'Original')
    mean = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
    std = plt.plot(rolstd, color = 'black',label = 'Rolling STd')
    plt.legend(loc = 'best')
    plt.title('To the Game')
    plt.show(block = False)
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[ ]:




# In[24]:

testStationary(ts)


# In[25]:

test = ts.rolling(window=12,center=False).mean()


# In[16]:

ts_log = np.log(ts)
plt.plot(ts_log)
plt.show(block = False)


# In[ ]:




# In[ ]:




# In[27]:

movingAverage = ts_log.rolling(window = 12,center = False).mean()
plt.plot(ts_log)
plt.plot(movingAverage, color = 'red')
plt.show()


# In[29]:

diff = ts_log - movingAverage
diff.head(12)


# In[32]:

diff.dropna(inplace = True)
testStationary(diff)


# In[34]:

movingAverage = ts_log.rolling(window = 6,center = False).mean()
plt.plot(ts_log)
plt.plot(movingAverage, color = 'red')
plt.show()


# In[35]:

diff = ts_log - movingAverage
diff.head(12)


# In[36]:

diff.dropna(inplace = True)
testStationary(diff)


# In[38]:

expweighted_avg = ts_log.ewm(halflife=12 , min_periods=0,adjust=True,ignore_na=False).mean()
plt.plot(ts_log)
plt.plot(expweighted_avg, color='red')


# In[39]:

diff2 = ts_log - expweighted_avg
diff2.dropna(inplace = True)
testStationary(diff2)


# In[17]:

ts_log_diff = ts_log - ts_log.shift()


# In[18]:

plt.plot(ts_log_diff)


# In[20]:

ts_log_diff.dropna(inplace = True)
testStationary(ts_log_diff)


# In[21]:

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# In[22]:

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition1 = seasonal_decompose(ts_log_diff)

trend1 = decomposition1.trend
seasonal1 = decomposition1.seasonal
residual1 = decomposition1.resid

plt.subplot(411)
plt.plot(ts_log_diff, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend1, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal1,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual1, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# In[23]:

ts_decompose = residual
ts_decompose.dropna(inplace = True)
testStationary(ts_decompose)


# In[24]:

from statsmodels.tsa.stattools import acf, pacf


# In[25]:

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')


# In[26]:

from statsmodels.tsa.arima_model import ARIMA


# In[27]:

model = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))


# In[28]:

model = ARIMA(ts_log, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))


# In[29]:

model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))


# In[31]:

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())


# In[33]:

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print (predictions_ARIMA_diff_cumsum.head())


# In[34]:

predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()


# In[35]:

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



