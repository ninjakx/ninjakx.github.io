---
layout: post
title: "Introduction to Time Series Analysis using Python"
featured-img: bgtsf
categories: [Machine learning]
---

# THE BASICS OF TIME SERIES ANALYSIS

## WHAT ARE TIME SERIES? 

Time series  is a series of data points indexed (or listed or graphed) in time order. 
Or in other words, a collection of observations of well-defined data items obtained through repeated measurements over time at equal intervals e.g hourly, daily, weekly, quarterly, yearly, etc. It is mostly used to predict future occurrences based on previous observed occurrence or values.

**Examples:**
* Continuous monitoring of a person’s heart rate,
* Hourly readings of temperature or humidity,
* Daily closing stock prices,
* Monthly precipitation in a specific location,
* Yearly sales revenue/figures.

![examples](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/TSEXAMPLE.png)

<p style='color:red'> NOTE: Data collected irregularly are not time series.</p>


## WHY DO WE NEED TIME SERIES?

This is the question that arises in our mind Why do we need to go for times series data?. The answer is very simple as we want to understand the past trends so we can plan for the future.
For example, How do you decide when is the right time to buy something. For instance, take the example of gold. We invest in the gold when we know its price is going to be high in the future but how do we come up with this?. Simply we just look at the historical sales data of it to draw some inference.


## COMPONENTS OF TIME SERIES:

* **Trend component** - a long-term increase or decrease in the data regularly through time which might not be linear. Sometimes the trend might change direction as time increases and result in varying mean over time.
It is the result of influences such as population growth, price inflation, and general economic changes. There can be Uptrend, Downtrend, Horizontal Trend depending on the pattern.

![trend example](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/trenddef.png)

* **Cyclical component** - exists when data exhibit rises and falls that are not of the fixed period. Here the average length of cycles is longer than the length of a seasonal pattern. 
In practice, the trend component is assumed to include also the cyclical component. Sometimes the trend and cyclical components together are called as trend-cycle.These kinds of patterns are much harder to predict. 

![cyclic example](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/2_cycl4.jpg)

* **Seasonal component** - exists when a series exhibits regular fluctuations based on the season (e.g. every month/quarter/year) and results in varying mean over time. Seasonality is always of a fixed and known period. It is the result of influences such as weather conditions, customs of people etc.

* **Irregular component** – It is the residual time series after the trend-cycle and the seasonal components (including calendar effects) have been removed. It corresponds to the high-frequency fluctuations of the series and is a stationary process.

![all in one example](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/ss.PNG)

## STATIONARITY

A stationary time series is one whose statistical properties such as mean, variance, autocorrelation, etc. are all constant over time. 
As Trends can result in a varying mean over time while that of seasonality can result in a changing variance over time. Thus time series with trends, or with seasonality, is not stationary.
But time series with cyclic behavior (but with no trend or seasonality) is stationary as the cycles are not of a fixed length, we cannot be sure where the peaks and troughs of the cycles will be.

![all the time series type](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/nst.png)

**Let us see which of the following series are stationary.**
<table class="tg">
  <tr>
    <td class="tg-0pky"><img src="https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/st1.png" width="600" height="300"></td>
    <td class="tg-0pky"><span style="color:red">Non Stationary</span><br>as there is a trend and seasonality(variance is also increasing).</td>
  </tr>
  <tr>
    <td class="tg-0pky"><img src="https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/st2.png" width="600" height="300"></td>
    <td class="tg-0pky"><span style="color:red">Non Stationary</span><br> as there is a Seasonality.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><img src="https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/st3.png" width="600" height="300"></td>
    <td class="tg-0pky"><span style="color:green">Stationary.</span><br> It has cycles which are aperiodic. In the long-term, the timing of these cycles is not predictable. Hence the series is stationary.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><img src="https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/st5.png" width="600" height="300"></td>
    <td class="tg-0pky"><span style="color:green">Stationary.</span><br></td>
  </tr>
  <tr>
    <td class="tg-0pky"><img src="https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/st6.png" width="600" height="300"></td>
    <td class="tg-0pky"><span style="color:red">Non Stationary</span><br> as there are Trends and changing levels.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><img src="https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/st7.png" width="600" height="300"></td>
    <td class="tg-0pky"><span style="color:red">Non Stationary</span><br> as there are Trends and changing levels.</td>
  </tr>
</table>

## MAKING A TIME SERIES STATIONARY

### DIFFERENCING

Differencing is a method of transforming a time series dataset.
`Differencing can help stabilize the mean of the time series by removing changes in the level of a time series, and so eliminating (or reducing) trend and seasonality.`

It is performed by subtracting the previous observation from the current observation. 

**difference(t) = observation(t) – observation(t-1)**

Once the prediction is made, we have to convert it back into its original scale.
To invert it back, we add the observation at the prior time step to the difference value.

**inverted(t) = differenced(t) + observation(t-1)**

Some terms you need to be aware of:

**Lag Difference**

The lag-1 difference is the difference between consecutive observations. In other words, you shift the time series by 1 and then take the difference of the observation at t and observation at t-1(shifted time series).
For time series with a seasonal component, the lag may be expected to be the period (width) of the seasonality. 

**Difference Order**

It is defined as the number of times differencing is performed until all temporal dependence has been removed.

### SEASONAL DIFFERENCING

In this instead of calculating the difference between consecutive values, we calculate the difference between an observation and a previous observation from the same season. 

### TRANSFORMATION

It is usually for making the time series stationary on variance. We can perform power transform, log transform or square root transform on series and then we do differencing to make the data stationary.


## DECOMPOSITION OF NON STATIONARY TIME SERIES

Non-stationary time series can have multiplication decomposition as well as additive decomposition.

### Additive decomposition:

In this, we assume that the different components affected the time series additively.

**Time series = Seasonal Effect + Trend + Cyclical + Residual**

A good example of additive time series is beer production.
Consider the quarterly beer production dataset.

```python
from pandas import Series
from matplotlib import pyplot
df = pd.read_csv('quarterly-beer-production-in-aus.csv',delimiter=',')
df.index = df['Quarter']
df.rename(columns = {df.columns[1]: 'Quartely_beer_production'}, inplace = True)
df.__delitem__('Quarter')
df.drop(df.index[len(df)-1],inplace=True)
df.index = pd.to_datetime(df.index) + pd.offsets.QuarterEnd(0)
df.plot()
```

![BEER DATASET](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/eer.png)

As we can see seasonality remains relatively constant only upward trend can be observed.
Let us decompose this time series data as a additive model.

```python
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
series = df
series.plot()
result = seasonal_decompose(series, model='additive')
result.plot()
pyplot.show()
```

![BEER Decompose](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/beer1.png)

### Multiplicative decomposition:

In this, we assume that the different components affected the time series proportionally.

**Time Series = Seasonal Effect * Trend * Cyclical * Residual**

If we want to fit the multiplicative model. We can take the Log on both the side of the above expression. 

**log(Time Series) = log(Seasonal Effect) + log(Trend) +  log(Cyclical)  + log( Residual)**

After then we will take the exponential of it.
A good example of multiplicative time series is Airline Passenger Numbers 

Consider the Airline Passengers dataset.

```python
from pandas import Series
from matplotlib import pyplot
df = pd.read_csv('AirPassengers.csv')
df.plot()
pyplot.show()
```

![Passenger Dataset](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/ps1.png)

Let us decompose this time series data as a multiplicative model.

``` python
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
series = Series.from_csv('AirPassengers.csv', header=0)
result = seasonal_decompose(series, model='multiplicative')
result.plot()
pyplot.show()
```

![Passenger Decompose](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/mu.png)

## ESTIMATING THE TREND

There are Two approaches:

1)Using a smoothing procedure.

2)Specifying a regression equation for the trend.

## 1)Using a smoothing procedure
We can apply Moving Average Filtering or Exponential Weighted Moving Average (EWMA) method.

**Moving  Average Filtering**

a moving average (rolling average or running average) is a calculation to analyze data points by creating a series of averages of different subsets of the full data set. 
It is used to smooth out short-term fluctuations and highlight longer-term trends or cycles. The threshold between short-term and long-term depends on the application, and the parameters of the moving average will be set accordingly. It identifies the trend directions.  A rising Moving Average indicates that the time series is in an uptrend, while a declining Moving Average indicates that it is in a downtrend. 

A moving average of order m can be written as:

![img](https://latex.codecogs.com/gif.latex?\hat{T}_{t}&space;=&space;\frac{1}{m}&space;\sum_{j=-k}^k&space;y_{t&plus;j})

The two commonly used Moving Averages are the Simple Moving Average (SMA) and the Exponential Moving Average (EMA).

**a)Simple Moving Average**

Simple Moving Average technical analysis indicator that averages quantity over a period of time and plots that average as a line.

```python
import matplotlib.pyplot as plt
import pandas as pd
x = pd.Series(np.arange(50))
time = pd.date_range('2014-01-01', periods=50,freq='m')
y = pd.Series(10 + (x + 2.5*np.random.randint(-5, + 20, 50)))
df1 = pd.DataFrame(y, columns=['sales_val'])
df1.index = time
df1.head(10)
```
![sales val](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/s1.PNG)

```python
# Calculating the short-window simple moving average
short_rolling = df1.rolling(window=10).mean()
short_rolling.head(10)
```

![sales val rolling](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/s2.PNG)

![img6](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Clarge%20S%28t%3D10%29%20%3D%20%5Cfrac%7BF%28t%3D0%29&plus;F%28t%3D1%29&plus;....&plus;F%28t%3D10%29%7D%7B10%7D%20%5C%5C%5C%5C%20%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%3D%20%5Cfrac%7B115&plus;105.4&plus;75.8&plus;71.2&plus;106.6&plus;82.0&plus;47.4&plus;117.8&plus;103.2&plus;83.6%7D%7B10%7D%5C%5C%20%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%20%3D%20%5Cfrac%7B908%7D%7B10%7D%20%5C%5C%20%5C%5C%20%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%20%3D%2090.8)

```python
# Calculating the long-window simple moving average
long_rolling = df1.rolling(window=70).mean()
#long_rolling.head(20)
```

```python
import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(16,5))

ax.plot(df1, label='sales_val')
ax.plot(long_rolling, label = '70-days SMA')
ax.plot(short_rolling, label = '10-days SMA')
ax.legend(loc='best')
ax.set_ylabel('Price in $')
my_year_month_fmt = mdates.DateFormatter('%m/%y')
ax.xaxis.set_major_formatter(my_year_month_fmt)
```

![rolling](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/s3.PNG)
We can see SMA time series are much less noisy than the original price time series but this SMA time series lag the original price time series with the lag(delay) of L months. 
In order to reduce the lag induced by SMA, we will go with the EMA. 

**b)Exponential Moving Average (EMA)**

![img3](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Clarge%20%5C%5C%20%5Ctext%7BEMA%7D%5Cleft%28t%5Cright%29%20%26%20%3D%20%5Cleft%281-%5Calpha%5Cright%29%5Ctext%7BEMA%7D%5Cleft%28t-1%5Cright%29%20&plus;%20%5Calpha%20%5C%20p%5Cleft%28t%5Cright%29%20%5C%5C%20%26%20%5Ctext%7BEMA%7D%5Cleft%28t_0%5Cright%29%20%26%20%3D%20p%5Cleft%28t_0%5Cright%29)

Where p(t) is the price at time t and α is the decay parameter of the EMA.

The relationship between α and that of lag is given as 

![img2](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Clarge%20%5Calpha%20%3D%20%5Cfrac%7B1%7D%7BL&plus;1%7D)

Or

![img4](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Clarge%20%5Calpha%20%3D%20%5Cfrac%7B2%7D%7BM%20&plus;%201%7D)

Where M is the length of the window/span.

EMA tend to reduce the lag as it puts more emphasis on recent observations.

```python
ema_short = df1.ewm(span=10,adjust=False).mean() # adjust is kept at false to enable the recursive calculation mode
ema_short.head(10)
```

![EMA](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/s11.PNG)

![img5](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Clarge%20%5C%5C%20%5Calpha%20%3D%20%5Cfrac%7B2%7D%7BM%20&plus;%201%7D%20%3D%20%5Cfrac%7B2%7D%7B10%20&plus;%201%7D%20%3D%200.1818%20%5C%5C%20%5C%5C%20E%28t%29%20%3D%20%281-%5Calpha%29*%20E%28t-1%29%20&plus;%20%5Calpha%20P%28t%29%20%5C%5C%20E%28t0%29%20%3D%20P%28t0%29%5C%5C%5C%5C%20E%28t%3D0%29%20%3D%20115%20%5C%5C%20E%28t%3D1%29%20%3D%20%281-0.1818%29E%28t%3D0%29&plus;0.1818P%28t%3D1%29%5C%5C%20%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%20%3D%200.818*115%20%7E&plus;%7E0.1818*105.4%5C%5C%20%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%20%3D%2094.0909%20%7E&plus;%7E19.1672%5C%5C%20%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%20%3D%20113.25)

```python
ema_long = df1.ewm(span=70,adjust=False).mean()
# adjust is kept at false to enable the recursive calculation mode
ema_long.head(10)
```

![ema_long](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/s12.PNG)

```python
fig, ax = plt.subplots(figsize=(16,5))

ax.plot(df1, label='Price')
ax.plot(ema_short, label = '10-days EMA')
ax.legend(loc='best')
ax.set_ylabel('Price in $')
my_year_month_fmt = mdates.DateFormatter('%m/%y')
ax.xaxis.set_major_formatter(my_year_month_fmt)
```

![combined](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/s4.PNG)

```python
import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(16,5))
df1['SMA_10days'] = short_rolling
df1['EMA_10days'] = ema_short
print(df1.head(20))
ax.plot(df1['sales_val'], label='sales_val')
ax.plot(short_rolling, label = '10-days SMA')
ax.plot(ema_short, label = '10-days EMA')

ax.legend(loc='best')
ax.set_ylabel('Price in $')
my_year_month_fmt = mdates.DateFormatter('%m/%y')
ax.xaxis.set_major_formatter(my_year_month_fmt)
df1.__delitem__('SMA_10days')
df1.__delitem__('EMA_10days')
```

![plot](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/scomb.PNG)

As we can see EMA reacts to price changes faster than the SMA.



### **ADVANTAGES OF MOVING AVERAGES** 
It can be used for linear as well as non-linear trends.


### **DISADVANTAGES OF MOVING AVERAGES**
The trend obtained by moving averages generally is neither a straight line nor a standard curve and hence can’t be extended for predicting future values. Trend value won’t be present for some periods at the start and at the end of the time series hence not applicable for short-term series.


## 2)Using Regression method 
We can use the linear model like linear regression if there are linear trends and for the nonlinear case, Polynomial or other curve-fitting method are used.

Let’s take the same time series data to fit a linear regression curve. 

```python
fig = plt.figure(figsize=(15,5))
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = [i for i in range(0, len(df1))]
X = np.reshape(X, (len(X), 1))
model.fit(X, y)
trend = model.predict(X)
# plot trend
pyplot.plot(y)
pyplot.plot(trend)
pyplot.show()
```

![reg](https://github.com/ninjakx/ninjakx.github.io/raw/master/assets/img/posts/trendreg.PNG)


## TESTING FOR STATIONARITY

### Augmented Dickey-Fuller test (ADF)

An augmented Dickey-Fuller test (ADF) tests the null hypothesis that a unit root(a feature that can cause issues in statistical inference) is present in a time series sample.
The Augmented Dickey-Fuller (ADF) statistic, used in the test, is a negative number. The more negative it is, the stronger the rejection of the hypothesis that there is a unit root at some level of confidence 
An autoregressive model is used and it tries to optimize information criterion across multiple different lag values. 

* **Null hypothesis(H0):** If accepted then the time series can be represented by a unit root which means it is not stationary i.e it has some time-dependent structure.
* **Alternate hypothesis(H1):** If the null hypothesis is rejected then the time series is stationary i.e It does not have a time-dependent structure. 
 
Depending on the p-value which we get from this test. We make the following two interpretations.
* **p-value>0.05** :Accept H0, the data has a unit root and is non-stationary.
* **p-value<=0.05**: Reject H0, the data does not have a unit root and is stationary.

```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(df2['sales_val'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
```

```
ADF Statistic: -0.667494
p-value: 0.854991
Critical Values:
        1%: -3.458       
        5%: -2.874       
        10%: -2.573
```

ADF Statistic is negative and lower but larger than the critical values.
the p-value is also above 0.05. We cannot reject H0. The series has a unit root and so it is non-stationary.


### Kwiatkowski–Phillips–Schmidt–Shin Test (KPSS)

It is carried out for testing a null hypothesis that an observable time series is stationary around a deterministic trend (i.e. trend-stationary) against the alternative of a unit root.
KPSS is based on linear regression. It breaks up a series into three parts: a deterministic trend (β<sub>t</sub>), a random walk (r<sub>t</sub>), and a stationary error (ε<sub>t</sub>), with the regression equation:  

![eq](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Clarge%20x_%7Bt%7D%20%3D%20r_%7Bt%7D%20&plus;%20%5Cbeta_%7Bt%7D%20&plus;%20%5Cvarepsilon_1)

If the KPSS statistic is greater than the critical values (at alpha levels of 10%, 5%, and 1%), then the null hypothesis is rejected; the series is non-stationary. 

```python
from statsmodels.tsa.stattools import kpss
result = kpss(df2['sales_val'])
#kpss_stat, p_value, crit_dict, rstore
print('KPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[3].items():
    print('\t%s: %.3f' % (key, value))
```

```
KPSS Statistic: 1.532254
p-value: 0.010000
Critical Values:
        10%: 0.347       
        5%: 0.463        
        2.5%: 0.574        
        1%: 0.739
```

Since the KPSS statistic is greater than the critical values, the Null hypothesis is rejected; the series is non-stationary.

<br>
<br>

## REFERENCES

[Moving averages](https://otexts.org/fpp2/moving-averages.html)

[Seasonal trend decomposition](https://anomaly.io/seasonal-trend-decomposition-in-r/)

[How to Check if Time Series Data is Stationary with Python](https://machinelearningmastery.com/time-series-data-stationary-python/)

[Moving average trading strategy](https://www.learndatasci.com/tutorials/python-finance-part-3-moving-average-trading-strategy/)

[Time series with trend and seasonality components](http://web.vu.lt/mif/a.buteikis/wp-content/uploads/2018/02/Lecture_03.pdf)

[KPSS Test: Definition and Interpretation](https://www.statisticshowto.datasciencecentral.com/kpss-test/)
























