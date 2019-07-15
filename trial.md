# Los Alamos National Laboratory Earthquake Prediction

#### Huy Thong Nguyen, Tung Hoang, Jordan Lightstone, Danial Huff
#### CX4240 Project

## I. Problem statements

Earthquakes are devastating natural disasters and possess the potential to destroy buildings and cities. In the process they can injure and even kill hundreds and thousands of people. It is necessary to develop solutions to better understand seismic activity. Futher more, it is a priority to develop tools capable of the accurate and precise prediction of when an earthquake will happen. Prediction tools of this nature could equip officials with the necessary information to advice about development plans and safty propocals long before these events occur. 

<p align="center">
  <img width="460" height="300" src="https://github.com/jlightstone/cs_project/blob/master/Graphs/earthquake.jpg">
</p>

Scientist have recently discovered that “constant tremors” measured along fault lines provide valuable information about earthquake activity. The goal of this project is to utilize a machine learning (ML) approach to analyze experimental data that simulates the tremors at fault lines. Ultimately, the goal of the project is to develop a generalized ML model capable of correctly predicting the time until failure.    

<p align="center">
  <img/ src="https://github.com/hoangtung167/cx4240/blob/master/CSV%20Files/Introduction_data.png">
</p>

The data above shows a continuous block of experimental data used for training and test our models. This graph shows plotted accoustic viabrations as a function of time. The plot also shows a "time to failure" component (a.k.a. the time till earthquake). In order to predict the time to failure, statistical features must be extracted from the data. The features used to describe the data are discussed in the next section. 

#### Environment Setup

<details><summary>CLICK TO EXPAND</summary>
<p>
  
```python
import os
from scipy import ndimage, misc
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_diabetes, load_digits, load_breast_cancer
from keras.datasets import mnist
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import statistics 
```
</p>
</details>

## II. Feature Extraction


![Feature Extraction Concept](https://github.com/hoangtung167/cx4240/blob/master/Graphs/Feature_Extraction_Concept.png)

Using resourses from [kaggle](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94390#latest-554034), we determined 16 statistical features that serve as potential candidates for understanding the experimental data. The features are broken into four different catagories- basic, fast fourier transformed, rolling window, and mel-frequency features. 

The 'basic' features are calculate using simple statistics and include ‘mean’, ‘std’, and ‘skew’.  The 'fast fourier transformed' features convert the time-domain signal into a frequency-domain signal, which results in real and imaginary numbers. The mean and standard deviation for the real and imaginary numbers are calculated, resulting in 4 additional features. The 'rolling windows method' resulted from breaking the larger data set into a rolling window size of 100. From this, the mean, standard deviation, and upper and lower percentiles subsets were used to extract an additional six features. Lastly, the Librosa toolbox was used to calculate the Mel-frequency cepstral coefficients of two features. Below four of the calculated features are plotted on the same graph as the time to failure. 

<p align="center">
  <img/ src="https://github.com/jlightstone/cs_project/blob/master/Graphs/Feature_1.png">
</p>
<p align="center">
  <img/ src="https://github.com/jlightstone/cs_project/blob/master/Graphs/Feature_2.png">
</p>

Following feature extraction, we train various ML algorithms. Our goal was to identify which algorthim is most capable of predicting time until failure. In the following section, we provide results pertaining to the prediction capabilities of a multitude of algorithms.

#### Feature Extractions Methods

<details><summary>CLICK TO EXPAND</summary>
<p>
  
```python
def generate_feature_basic(seg_id, seg, X):
    xc = pd.Series(seg['acoustic_data'].values)
    
    X.loc[seg_id, 'index'] = seg_id
    X.loc[seg_id, 'mean'] = xc.mean()
    X.loc[seg_id,'std'] = xc.std()
    #X.loc[seg_id, 'kurt'] = xc.kurtosis()
    X.loc[seg_id, 'skew'] = xc.skew()
def generate_feature_FFT(seg_id, seg, X):
    xc = pd.Series(seg['acoustic_data'].values)
    zc = np.fft.fft(xc)
    realFFT, imagFFT = np.real(zc), np.imag(zc)
    
    X.loc[seg_id, 'FFT_mean_real'] = realFFT.mean()
    X.loc[seg_id, 'FFT_mean_imag'] = imagFFT.mean()
    X.loc[seg_id, 'FFT_std_real'] = realFFT.std()
    X.loc[seg_id, 'FFT_std_max'] = realFFT.max()
    
def generate_feature_Roll(seg_id, seg, X):
    xc = pd.Series(seg['acoustic_data'].values)
    
    windows = 100
    x_roll_std = xc.rolling(windows).std().dropna().values
    x_roll_mean = xc.rolling(windows).mean().dropna().values
    
    X.loc[seg_id, 'Roll_std_p05'] = np.percentile(x_roll_std, 5)
    X.loc[seg_id, 'Roll_std_p30'] = np.percentile(x_roll_std,30)
    X.loc[seg_id, 'Roll_std_p60'] = np.percentile(x_roll_std,60)
    X.loc[seg_id, 'Roll_std_absDiff'] = np.mean(np.diff(x_roll_std))
    
    X.loc[seg_id, 'Roll_mean_p05'] = np.percentile(x_roll_mean, 5)
    X.loc[seg_id, 'Roll_mean_absDiff'] = np.mean(np.diff(x_roll_mean))

def generate_feature_Melfrequency(seg_id, seg, X):
    xc = seg['acoustic_data'].values
    mfcc = librosa.feature.mfcc(xc.astype(np.float64))
    mfcc_mean = mfcc.mean(axis = 1)
    
    X.loc[seg_id, 'MFCC_mean02'] = mfcc_mean[2]
    X.loc[seg_id, 'MFCC_mean16'] = mfcc_mean[16]

chunksize = 150000
CsvFileReader = pd.read_csv('train.csv', chunksize = chunksize)
X, y = pd.DataFrame(), pd.DataFrame()

for seg_id, seg in tqdm_notebook(enumerate(CsvFileReader)):
    y.loc[seg_id, 'target'] = seg['time_to_failure'].values[-1]
    generate_feature_basic(seg_id, seg, X)
    generate_feature_FFT(seg_id, seg, X)
    generate_feature_Roll(seg_id, seg, X)
    generate_feature_Melfrequency(seg_id, seg, X)

X.to_csv('extract_train_Jul08.csv')
y.to_csv('extract_label_Jul08.csv')
X1, y1 = X.iloc[500:1000], y.iloc[500:1000]

plt.figure(figsize=(15, 15))
for i, col in enumerate(X.columns):
    ax1 = plt.subplot(4, 4, i + 1)
    plt.plot(X1[col], color='blue');plt.title(col);ax1.set_ylabel(col, color='b')
    ax2 = ax1.twinx(); plt.plot(y1, color='g'); ax2.set_ylabel('time_to_failure', color='g')
    ax1.legend(loc= 2);ax2.legend(['time_to_failure'], loc=1)
plt.subplots_adjust(wspace=0.5, hspace=0.3)
```
</p>
</details>

### Feature Extractions for training data

Since the training data is a large csv file (9.5GB), which exceeds the computation capability of our laptop, we use the pandas with `chunksize = 150000` to load one time-series windown at one. At each time window, 150_000 input data is transformed into 16 dimensional vectors (16 features) and append to input dataframe. The target is the `time before failure` is also appended to a separate dataframe.

<details><summary>CLICK TO EXPAND</summary>
<p>
  
```python

chunksize = 150000
CsvFileReader = pd.read_csv('train.csv', chunksize = chunksize)
X, y = pd.DataFrame(), pd.DataFrame()

for seg_id, seg in tqdm_notebook(enumerate(CsvFileReader)):
    y.loc[seg_id, 'target'] = seg['time_to_failure'].values[-1]
    generate_feature_basic(seg_id, seg, X)
    generate_feature_FFT(seg_id, seg, X)
    generate_feature_Roll(seg_id, seg, X)
    generate_feature_Melfrequency(seg_id, seg, X)

X.to_csv('extract_train_Jul08.csv')
y.to_csv('extract_label_Jul08.csv')
```

</p>
</details>

The resulting table is as follows:

|    | index | mean     | std      | skew     | FFT_mean_real | FFT_mean_imag | FFT_std_real | FFT_std_max | Roll_std_p05 | Roll_std_p30 | Roll_std_p60 | Roll_std_absDiff | Roll_mean_p05 | Roll_mean_absDiff | MFCC_mean02 | MFCC_mean16 |
|----|-------|----------|----------|----------|---------------|---------------|--------------|-------------|--------------|--------------|--------------|------------------|---------------|-------------------|-------------|-------------|
| 0  | 0     | 4.884113 | 5.101106 | -0.02406 | 12            | -1.70E-15     | 2349.811     | 732617      | 2.475639     | 2.848551     | 3.367387     | -5.24E-06        | 4.16          | -2.40E-06         | -26.0598    | 5.51279     |
| 1  | 1     | 4.725767 | 6.588824 | 0.390561 | 5             | 1.84E-15      | 2566.032     | 708865      | 2.475965     | 2.847842     | 3.38893      | -4.87E-07        | 4.05          | -7.34E-07         | -26.4857    | 5.695142    |
| 2  | 2     | 4.906393 | 6.967397 | 0.217391 | 5             | 4.85E-16      | 2683.549     | 735959      | 2.538591     | 2.942616     | 3.589814     | 5.39E-06         | 4.14          | 5.07E-06          | -26.484     | 5.620199    |
| 3  | 3     | 4.90224  | 6.922305 | 0.757278 | 5             | -1.07E-15     | 2685.789     | 735336      | 2.496442     | 2.863141     | 3.442515     | -8.68E-06        | 4.16          | -5.34E-07         | -25.5651    | 5.241189    |


## III. Testing Maching Learning Models

### Linear Regression and Polynomial Regression Performance

Analysis began with linear and polynomial regression. With in the linear regression framework, ltiple methods were tested- Ridge, Lasso, and Huber Regressor. For the polynomial model we fit the model to a 2nd degree polynomial. 

The linear regression model provide fairly acceptable prediction for "time before failure". However, the model was unable to predict high peaks and yields a consistent trend of repeating height - nearly periodic. The polynomial regression model yields nearly identical resuts to the linear regression model, but displays a slightly larger error.

From the graphs below, we can observe that the Mean Absolute Error has a tendency to increase as the polynomial degree is increased. Furthermore, the variance increases significantly as the polynomial degree is increased. This likely indicates that overfitting has occured. Ultimately we decide to use 2nd degree polynomial to build a model (1st degree polynomial is simply a linear function). 

<p align="center">
  <img/ src="https://github.com/hoangtung167/cx4240/blob/master/Graphs/Compare%20Predicted%20Values.png">
</p>
<p align="center">
  <img/ src="https://github.com/hoangtung167/cx4240/blob/master/Graphs/Compare%20MAE%20Linear%20Polynomial.png">
</p>

To yield better results, we turned to more complex modeling methods: Support Vector Machines, Neural Networks, Decision Trees, Random Forest, and LGB Classifier. These models are addressed below.  

#### Linear and Polynomial Regression Analysis
<details><summary>CLICK TO EXPAND</summary>
<p>
  
##### Comparing Linear and Polynomial Regression
<details><summary>CLICK TO EXPAND</summary>
<p>

```python

#creating models and plots

reg = LinearRegression().fit(features, target)

indx = range(target.shape[0])
plt.axis([0, target.shape[0], -0.1, 16])
plt.title("Comparison - Linear Regression")
plt.ylabel("Time before failure(s)")
plt.xlabel("Index")
plt.plot(indx, reg.predict(features), linewidth = 3, label = 'Pred')
plt.plot(indx, target, linewidth = 2, label = 'Actual')
plt.legend(loc='upper right')
plt.savefig('Compare P Linear.png', dpi = 324)
plt.show()

poly = PolynomialFeatures(degree=2)
features_poly = poly.fit_transform(features)
reg = LinearRegression().fit(features_poly, target)

indx = range(target.shape[0])
plt.axis([0, target.shape[0], -0.1, 16])
plt.title("Comparison - Polynomial Regression")
plt.ylabel("Time before failure(s)")
plt.xlabel("Index")
plt.plot(indx, reg.predict(features_poly), linewidth = 3, label = 'Pred')
plt.plot(indx, target, linewidth = 2, label = 'Actual')
plt.legend(loc='upper right')
plt.savefig('Compare P Polynomial.png', dpi = 324)
plt.show()

#calculating error
fl = ['Linear Regression', 'Polynomial Regression']
t1, m1, v1, c1 = K_Fold(features,target, degree = 1, numfolds = 5, classifier = "linear")
t2, m2, v2, c2 = K_Fold(features,target, degree = 2, numfolds = 5, classifier = "polynomial")
mae = np.append(m1, m2)
var = np.append(v1,v2)
## coeff = reg.coef_.shape
materials = fl
x_pos = np.arange(len(fl))
CTEs = mae
error = var
# Build the plot
fig, ax = plt.subplots(1,2,figsize =(9,3))
ax[0].bar(x_pos, CTEs, yerr=error, align='center', color = ['red', 'green'], alpha=0.5, ecolor='black', capsize=10)
ax[0].set_ylim(0, 3)
ax[0].set_ylabel('Mean Absolute Error')
ax[0].set_xticks(x_pos)
ax[0].set_xticklabels(materials)
ax[0].set_title('Kfold results')
ax[0].yaxis.grid(True)

CTEs = [2.110853811043013, 1.985654086901071]

ax[1].bar(x_pos, CTEs, align='center', color = ['red', 'green'], alpha=0.5, ecolor='black', capsize=10)
ax[1].set_ylim(0, 3)
ax[1].set_ylabel('Mean Absolute Error')
ax[1].set_xticks(x_pos)
ax[1].set_xticklabels(materials)
ax[1].set_title('Training the whole set')
ax[1].yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('Compare MAE Linear Polynomial.png', dpi = 199)
plt.show()
```

</p>
</details>

##### Linear Regression 
<details><summary>CLICK TO EXPAND</summary>
<p>

```python
target = pd.read_csv("extract_label_Jul08.csv", delimiter = ',')
target = target.as_matrix()
target = target[:,1]

features = pd.read_csv("extract_train_Jul08.csv", delimiter = ',')
features = features.as_matrix()
features = features[:, 1:17]

def K_Fold(features, target, degree, numfolds, classifier):
    numfolds += 1
    kf = KFold(n_splits=numfolds)
    kf.get_n_splits(features)

    i = 0
    mae = np.zeros(numfolds-1)
    coef = np.zeros([numfolds-1, 16])
    for train_index, test_index in kf.split(features):
        features_train, features_test = features[train_index], features[test_index]
        target_train, target_test = target[train_index], target[test_index]

        poly = PolynomialFeatures(degree)
        features_poly_train = features_train
        features_poly_test = features_test

        if classifier == "polynomial" :
            features_poly_train = poly.fit_transform(features_train)
            features_poly_test = poly.fit_transform(features_test)

        reg = LinearRegression().fit(features_poly_train, target_train)
        
        if classifier == "ridge" :
            clf = Ridge(alpha=0.001)
            reg = clf.fit(features_poly_train, target_train)
        if classifier == "linear":
            if (i < numfolds - 1):
                coef[i, :] = reg.coef_.reshape(1, 16)
        if classifier == "lasso":
            clf = linear_model.Lasso(alpha=0.1)
            reg = clf.fit(features_poly_train, target_train)
        if classifier == "huber":
            reg = HuberRegressor().fit(features_poly_train, target_train)

        i = i+1
        if (i < numfolds):
            mae[i-1] = mean_absolute_error(target_test, reg.predict(features_poly_test))

        avrmae = (sum(mae)/(numfolds-1))
        var = (statistics.variance(mae))
        return mae, avrmae, var, coef

def mv(coefmat):
    
    mean = np.zeros(coefmat.shape[1])
    var = np.zeros(coefmat.shape[1])
    for i in range(coefmat.shape[1]):
        mean[i] = np.mean(coefmat[:, i])
        var[i] = np.std(coefmat[:, i])
    return mean, var

reg = LinearRegression().fit(features, target)
print("The loss values is: ", mean_absolute_error(target, reg.predict(features)))
indx = range(target.shape[0])
plt.axis([0, target.shape[0], -0.1, 16])
plt.title("Comparison between predicted and actual target values")
plt.ylabel("Time before failure(s)")
plt.xlabel("Index")
plt.plot(indx, reg.predict(features), linewidth = 3, label = 'Pred')
plt.plot(indx, target, linewidth = 2, label = 'Actual')
plt.legend(loc='upper right')
plt.savefig('Linear Regression.png', dpi = 199)

#different linear regression types 
fl = ['Linear', 'Ridge', 'Lasso', 'Huber Regressor']
## coeff = reg.coef_.shape
materials = fl
x_pos = np.arange(len(fl))
t1, m, v, c = K_Fold(features,target,degree = 1, numfolds = 5, classifier = "linear")
t2, m1, v1, c1 = K_Fold(features,target,degree = 1, numfolds = 5, classifier = "ridge")
t3, m2, v2, c2 = K_Fold(features,target,degree = 1, numfolds = 5, classifier = "lasso")
t4, m3, v3, c3 = K_Fold(features,target,degree = 1, numfolds = 5, classifier = "huber")
tot = np.append(np.append(np.append(t1,t2), t3), t4)
mae = [m, m1, m2, m3]
var = [v, v1, v2, v3]
CTEs = mae
error = var

# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', color = ['black', 'red', 'green', 'blue', 'cyan'], alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Mean Absolute Error')
ax.set_xlabel('Types of Regressor')
ax.set_xticks(x_pos)
ax.set_xticklabels(materials)
ax.set_title('KFold Mean Absolute Error Values with Variances')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('Linear Regression K Fold.png', dpi = 199)
plt.show()

#feature importance
fl = ['index', 'mean', 'std', 'skew', 'FFT_mean_real', 'FFT_mean_imag',
     'FFT_std_real', 'FFT_std_max', 'Roll_std_p05', 'Roll_std_p30',
      'Roll_std_p60', 'Roll_std_absDiff', 'Roll_mean_p05',
      'Roll_mean_absDiff', 'MFCC_mean02', 'MFCC_mean16']
t, m, v, c = K_Fold(features,target, degree = 1, numfolds = 5, classifier = "linear")
mean, error = mv(c)
## coeff = reg.coef_.shape
materials = fl
x_pos = np.arange(len(fl))
CTEs = -mean/ np.amax(-mean)
error = error/ np.amax(-mean)
# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', color = ['black', 'red', 'green', 'blue', 'cyan'], alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Gradient value of features')
ax.set_xticks(x_pos)
ax.set_xticklabels(materials, rotation = 'vertical')
ax.set_title('Features Importance')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png', dpi = 199)
plt.show()
```
</p>
</details>

##### Polynomial Regression 
<details><summary>CLICK TO EXPAND</summary>
<p>
  
```python
  
fl = ['1','2','3','4']
i = np.array([1,2,3,4])
## coeff = reg.coef_.shape
materials = fl
x_pos = np.arange(len(fl))
tot = np.zeros(1)
mae = np.zeros(i.shape[0])
var = np.zeros(i.shape[0])
for numfold in range(i.shape[0]):
    t, m, v, c = K_Fold(features,target, degree = i[numfold], numfolds = 5, classifier = "polynomial")
    mae[numfold] = m
    var[numfold] = v
    tot = np.append(tot, t, axis = 0)
CTEs = mae
error = var
# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', color = ['black', 'red', 'green', 'blue', 'cyan'], alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Mean Absolute Error')
ax.set_xlabel('Degree Levels')
ax.set_xticks(x_pos)
ax.set_xticklabels(materials)
ax.set_title('KFold Mean Absolute Error Values with Variances')
ax.yaxis.grid(True)
tot = np.delete(tot, 0)
# Save the figure and show
plt.tight_layout()
plt.savefig('Polynomial K Fold.png', dpi = 199)
plt.show()

#fitting 2nd degree olynomial model 
poly = PolynomialFeatures(degree=2)
features_poly = poly.fit_transform(features)
reg = LinearRegression().fit(features_poly, target)
print("The loss values is: ", mean_absolute_error(target, reg.predict(features_poly)))

#plotting prediction results 
indx = range(target.shape[0])
plt.axis([0, target.shape[0], -0.1, 16])
plt.title("Comparison between predicted and actual target values")
plt.ylabel("Time before failures")
plt.xlabel("Index")
plt.plot(indx, reg.predict(features_poly), linewidth = 3, label = 'Pred')
plt.plot(indx, target, linewidth = 2, label = 'Actual')
plt.legend(loc='upper right')
plt.savefig('Polynomial Regression.png', dpi = 199)
```  
</p>
</details>

</p>
</details>

### Support Vector Machine and Neural Network

### Nerual Net (NN)

NN Validation (no_index) MeanAbsoluteError: Mean = 2.113 Std = 0.033

NN Validation (index) MeanAbsoluteError: Mean = 2.071 Std = 0.034
![NN with Index](https://github.com/hoangtung167/cx4240/blob/master/Graphs/NN_with_index.png)

### Support Vector Machine (SVM)
  
SVM Validation (index) MeanAbsoluteError: Mean = 2.065 Std = 0.038
![SVM linear with Index](https://github.com/hoangtung167/cx4240/blob/master/Graphs/SVM_linear_withIndex.png)

#### Building NN and SVM

<details><summary>CLICK TO EXPAND</summary>
<p>
  
```python
#environment set up
from keras.models import Sequential
from keras.layers import Dense

from sklearn.svm import SVR
from sklearn.feature_selection import RFE

#building NN
model = Sequential()
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

#support vector machine
model = SVR(kernel='linear')
model = SVR(kernel='rbf')

```
</p>
</details>

### Decision Tree/ Random Forest / LGB Classifier

We use 3 different types of Tree-Classifier methods including Decision Tree, Random Forest, and LGB classifier. We build two models with each model and used a five fold cross validation framework. The first set of models use all 16 features while the second set of models do not account for the feature 'index' (time dependent variable). In the graph below, it is clear that the LGB model with the 'index' feature performs the best. It has a MAE over 2x smaller than the next best model. 

<p align="center">
  <img/ src="https://github.com/hoangtung167/cx4240/blob/master/Graphs/Tree_score.png">
</p>

The results from the LGB model with the 'index' feature is displayed below. By simply doing a visual check one can already tell there is a significant improvement from the polynomial and linear regression models. In particular this model is now able to account for nonuniformities in peak height. Additionally, the important features are shown in the bar graph. It emphasized the importance of knowing time component of the singal under analysis. 

Validation MeanAbsoluteError: Mean = 0.680 Std = 0.036

<p align="center">
  <img/ src="https://github.com/hoangtung167/cx4240/blob/master/Graphs/LGBM_withIndex.png">
</p>

<details><summary>CLICK TO EXPAND</summary>
<p>
  
```python
#import packages
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

#decision tree model
model = DecisionTreeRegressor(min_samples_split = 25, random_state = 1, 
                                  criterion='mae',max_depth=5)

#random forrest
model = RandomForestRegressor(max_depth=5,min_samples_split=9,random_state=0,
                                  n_estimators=50,criterion='mae')                                
```
</p>
</details>

### Identifying the Best Model

In the figure below the MAE score with five-fold cross validation is reported for every model build during our prelimiary steps. It is clear that the LGBM (with index) model results in the lowest MAE. Based upon this analysis, moving forward it our goal to optimize the performance of this model. We are going to use PCA to reduce the dimensionality of our data set. It is our belief that reducing the dimensionality of our dataset will result in a increase performance of the LGB model. PCA work is addressed in the following section. 

#### Compare the loss values and variances across methods

<p align="center">
  <img/ src="https://github.com/hoangtung167/cx4240/blob/master/CSV%20Files/Summary_MAE_Score.png">
</p>



## V. Principal Component Analysis - PCA

Principal component analysis (PCA) is a technique used for understanding the dimensional structure of a data set. PCA transforms data in a way that converts a set of orthogonally correlated observations into a set of linearly uncorrelated variables called principal components.  This transformation maximizes the largest possible variance between each principal component and is a technique used to maximize the performance of a model.

In this work we use three different visualization methods to help understand the dimensional structure of the data and reduce the dimensionality of the dataset using PCA. 

### Pricipal Component Proportionality

The x-axis of the graph below indicated each principal component for the featurized data set, while the y-axis accounts for the proportionality of the total variance contained within the data set. As expected, the first principal component accounts for the largest amount of variance. Each consecutive principal component accounts for slightly less variance than component before it. 

The red line shows the cumulative proportional variance after each principal component is formed. The dashed line is an indication of 99% variance of the data. One can see that the dashed line crosses the cumulative sum (red) line at the 9th principal component. This indicated that 99% of the variance within the data is accounted for when the dimensionality of the data is reduced from 16 dimensions down to 9 dimensions. 

<p align="center">
  <img/ src="https://github.com/hoangtung167/cx4240/blob/master/Graphs/principal_component_visualization.png">
</p>

### Feature Variance for Principal Components 1 & 2

The two plots show the contributing variance of each feature in the first and second principal components. Yellow indicates a high positive variance while purple indicates a high negative variance. In the first principal component the features contributing to the most variance are the ‘Roll_std_pXX’ features as well as the “MFCC_mean02” feature. In the second principal component the “mean”, “FFT_std_max”, and “index” features contribute to the most variance. Knowing this correlation relationship could provide a framework for identifying the features providing the most significant variation within the model. 

<p align="center">
  <img/ src="https://github.com/hoangtung167/cx4240/blob/master/Graphs/first_principal_component.png">
</p>

<p align="center">
  <img/ src="https://github.com/hoangtung167/cx4240/blob/master/Graphs/second_principal_component.png">
</p>

### Visualizing Feature Correlation

The final graph within this section is a heat map which shows the correlation between different features. Dark red indicates that features have a strong positive correlation while dark blue indicates that there is a strong negative correlation between features. This heat map provides insight into which features are linearly independent and which variables linearly dependent. For example, the “Roll_std_p60” and “skew” features are linearly independent and have nearly zero correlation with other features. On the other hand, “Roll_std_60” is strongly correlated with 7 other features. 

<p align="center">
  <img/ src="https://github.com/hoangtung167/cx4240/blob/master/Graphs/heat_map.png">
</p>


Now that we have a better understanding of the feature importance and optimal dimensionality of our dataset, we plan to train a model on the reduced dimensionality matrix to see if we can improve our prediction capabilities. 

#### Generate PCA Model and Visualization
<details><summary>CLICK TO EXPAND</summary>
<p>
 
```python
#environment set up 
  import numpy as np
  import seaborn as sns
  import matplotlib.pyplot as plt
  import pandas as pd
  from sklearn.preprocessing import StandardScaler
  from sklearn.decomposition import PCA
  
#import data 
  train = pd.read_csv('extract_train_Jul08.csv')
  train = train.drop(['index'], axis = 1)
  train = train.drop(train.columns[0],axis = 1)

#standardize data and fit PCA model
  scaler=StandardScaler() #instantiate
  scaler.fit(train) # compute the mean and standard which will be used in the next command
  X_scaled=scaler.transform(train)
  pca=PCA() 
  pca.fit(X_scaled) 
  X_pca=pca.transform(X_scaled)
  ex_variance=np.var(X_pca,axis=0)
  ex_variance_ratio = ex_variance/np.sum(ex_variance)
  print(ex_variance_ratio)

#plotting PC variance proportion summation 
plt.figure(figsize=(10,5))
plt.bar(np.arange(1,16),pca.explained_variance_ratio_, linewidth=3)
plt.plot(np.arange(1,16),np.cumsum(pca.explained_variance_ratio_), linewidth=3, c = 'r', label = 'Cumulative Proportion')
plt.legend()
plt.xlabel('Principal Component')
plt.ylabel('Variance Proportion')
plt.grid()
plt.plot([0.99]*16, '--')
ex_variance=np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio)

#plot feature variance within 1st and 2nd principle component

plt.matshow([pca.components_[0]],cmap='viridis')
plt.yticks([0],['1st Comp'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(train.columns)),train.columns,rotation=65,ha='left')
plt.show()

plt.matshow([pca.components_[1]],cmap='viridis')
plt.yticks([0],['2nd Comp'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(train.columns)),train.columns,rotation=65,ha='left')
plt.show()

#feature correlation map
features = test.columns
plt.figure(figsize=(8,8))
s=sns.heatmap(test.corr(),cmap='coolwarm') 
s.set_yticklabels(s.get_yticklabels(),rotation=30,fontsize=7)
s.set_xticklabels(s.get_xticklabels(),rotation=30,fontsize=7)
plt.show()

a = np.abs(pca.components_[0])
a = a/np.max(a)
df = pd.DataFrame()
df['features'] = test.columns
df['importance'] = a
df.to_csv('PCA_extracted.csv')
print(df.shape)

#exporting feature importance
pca=PCA(n_components = 9) 
pca.fit(X_scaled) 
X_pca=pca.transform(X_scaled)
df = pd.DataFrame(X_pca)
df.to_csv('pca_exported_9features.csv')
```
</p>
</details>

### PCA with LGBM

Since we identify the LBGM achieves the highest 5fold Cross Validation score, we apply PCA with dimensionality reduction on the dataset.
Clearly shown on the graph is that we achieve similar performance with dimension = 10 compared to dimension = 16.

![LGBM_PCA](https://github.com/hoangtung167/cx4240/blob/master/Graphs/LGBM_PCA.png)


## VI. Summary

#### Comparing the feature importance across methods


<p align="center">
  <img/ src="https://github.com/hoangtung167/cx4240/blob/master/CSV%20Files/Summary_Feature_Importance.png">
</p>


## VIII. References

[1]	Rouet‐Leduc et al. Machine learning predicts laboratory earthquakes. Geophysical Research Letters, (2017) https://doi.org/10.1002/2017GL074677

[2]	Claudia Hulbert et al. Similarity of fast and slow earthquakes illuminated by Machine Learning, Nature Geoscience (2018). DOI: 10.1038/s41561-018-0272-8 

[3]	https://www.kaggle.com/c/LANL-Earthquake-Prediction/data

[4]	Breiman, L. (2001). Random forests. Machine Learning, 45(1), 532. https://doi.org/10.1023/A:1010933404324
