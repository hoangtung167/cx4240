## I. Problem statements
**Please edit this part**

Scientists at Los Alamos Laboratory have recently found a use for massive amounts of data generated by a “constant tremor” of fault lines where earthquakes are most common **[1-3]**. This data has previously been disregarded as noise. However, now, it has been proven useful through the lens of Machine Learning (ML) **[1-2]**. Following their recent publications, our goal is to build _Machine Learning regression models for the Laboratory Earthquake problem_ that if applied to real data, might help predict earthquakes before they happen

#### Data preview
**show 1% of the data**

#### Feature Extraction input
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


From the 150_000 acoustic data containing “random” number, we transform this entire time-series window (each has 150_000 data) into 16 statistical features. The features is selected based on the following public release:
[link1](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94390#latest-554034)
[link2](https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction)

### Feature definitions

**Basic features (4 features- ‘Index’, ‘mean’, ‘std’, ‘skew’):**  
From 150_000 data, we report the time when the signal is recorded (‘Index’), a single mean value (‘mean’), standard deviation (‘std’), and skew (‘skew’).

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
```

</p>
</details>

**Fast Fourier Transform (4 features:‘FFT_mean_imag’, ‘FFT_mean_real’, ‘FFT_std_max’, ‘FFT_std_real’):**  
Transform the time-domain signal into frequency-domain signal. Since it is complex number in the frequency-domain, I separate them into real and imaginary parts, each is reported with its mean and standard deviation

<details><summary>CLICK TO EXPAND</summary>
<p>
  
```python
def generate_feature_FFT(seg_id, seg, X):
    xc = pd.Series(seg['acoustic_data'].values)
    zc = np.fft.fft(xc)
    realFFT, imagFFT = np.real(zc), np.imag(zc)
    
    X.loc[seg_id, 'FFT_mean_real'] = realFFT.mean()
    X.loc[seg_id, 'FFT_mean_imag'] = imagFFT.mean()
    X.loc[seg_id, 'FFT_std_real'] = realFFT.std()
    X.loc[seg_id, 'FFT_std_max'] = realFFT.max()
```

</p>
</details>

**Rolling windows: (6 features:‘Roll_mean_absDiff’,‘Roll_mean_p05’,‘Roll_std_absDiff’,‘Roll_std_p05’,‘Roll_std_p30’,‘Roll_std_p60’)**  

From the 150_000 data, we choose a rolling window size = 100, at each window, we calculate the mean and standard deviation of each window. We use the numpy percentile function to calculate 5%,30%, 60% of the standard deviation (‘Roll_std_p05’,‘Roll_std_p30’,‘Roll_std_p60’), the top 5% of the mean, the mean of the gradient of these vectors (‘Roll_mean_absDiff’, ‘Roll_std_absDiff’).


<details><summary>CLICK TO EXPAND</summary>
<p>
  
```python
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
```

</p>
</details>

**Mel-frequency cepstral coefficients (2 features:‘MFCC_mean02’ ‘MFCC_mean16’)**
We use the Librosa toolbox to calculate the Mel-frequency cepstral coefficients of the 2nd and 16th components. 

<details><summary>CLICK TO EXPAND</summary>
<p>
  
```python
def generate_feature_Melfrequency(seg_id, seg, X):
    xc = seg['acoustic_data'].values
    mfcc = librosa.feature.mfcc(xc.astype(np.float64))
    mfcc_mean = mfcc.mean(axis = 1)
    
    X.loc[seg_id, 'MFCC_mean02'] = mfcc_mean[2]
    X.loc[seg_id, 'MFCC_mean16'] = mfcc_mean[16]
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
| 4  | 4     | 4.90872  | 7.30111  | 0.064531 | 12            | -1.46E-16     | 2761.716     | 736308      | 2.491521     | 2.863405     | 3.404453     | 1.54E-06         | 4.21          | 6.67E-07          | -24.8474    | 5.114833    |
| 5  | 5     | 4.913513 | 5.434111 | -0.1007  | 14            | 3.88E-16      | 2420.968     | 737027      | 2.518196     | 2.89498      | 3.404379     | -1.10E-05        | 4.19          | -4.34E-06         | -27.2839    | 5.498871    |
| 6  | 6     | 4.85566  | 5.687823 | 0.20881  | -4.46E-15     | -5.82E-16     | 2437.524     | 728349      | 2.506638     | 2.879464     | 3.463679     | -2.79E-06        | 4.19          | -1.60E-06         | -27.8118    | 5.953215    |
| 7  | 7     | 4.505427 | 5.854512 | -0.17633 | 3             | -1.33E-16     | 2361.259     | 675814      | 2.470411     | 2.811574     | 3.222396     | 2.19E-06         | 3.87          | 3.80E-06          | -24.9011    | 5.303046    |
| 8  | 8     | 4.717833 | 7.789643 | -0.16017 | 1             | -8.61E-16     | 2805.303     | 707675      | 2.572799     | 2.998636     | 3.747848     | -8.07E-06        | 4             | -1.80E-06         | -28.1274    | 5.274542    |
| 9  | 9     | 4.73096  | 6.890459 | 0.150779 | 5             | 4.85E-16      | 2620.174     | 709644      | 2.51611      | 2.915285     | 3.575718     | -4.13E-06        | 4.01          | 7.14E-06          | -27.4852    | 5.983643    |
| 10 | 10    | 4.582873 | 6.157272 | 1.572985 | 3             | 1.16E-15      | 2452.427     | 687431      | 2.483786     | 2.858197     | 3.409716     | 1.87E-07         | 3.87          | -5.47E-06         | -26.2505    | 5.390637    |

### Visualization of 16 features

<details><summary>CLICK TO EXPAND</summary>
<p>
  
```python
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

 ![Feature Visualization](https://github.com/hoangtung167/cx4240/blob/master/Graphs/Feature_Visualization.png)
 
 ### Load the test data and Visualize

The test data does not provide the information on the time when the data is recorded. We believe this information is encoded inside the segment ID in heximal form. Nonetheless, we can make the prediction with the rest of 15 features.

<details><summary>CLICK TO EXPAND</summary>
<p>
  
```python
submission = pd.read_csv('sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame()
plt.figure(figsize=(16,10))
plt.subplots_adjust(wspace=0.5, hspace=0.6)
for ii,seg_name in tqdm_notebook(enumerate(submission.index)):
    seg = pd.read_csv('test/{}.csv'.format(seg_name))
    generate_feature_basic(seg_name, seg, X_test)
    generate_feature_FFT(seg_name, seg,  X_test)
    generate_feature_Roll(seg_name, seg,  X_test)
    generate_feature_Melfrequency(seg_name, seg,  X_test)
    
    if ii<18:
        ax1 = plt.subplot(3, 6, ii + 1)
        plt.plot(seg['acoustic_data'].values, color='blue')
        plt.title(seg_name)
X.to_csv('extract_test_Jul08.csv')
```

</p>
</details>


 ![Test_Signal_Visualization](https://github.com/hoangtung167/cx4240/blob/master/Graphs/Test_set_visualization.png)

![Training data](https://github.com/hoangtung167/cx4240/blob/master/Dataset/extract_train_Jul08.csv)

To access original data set, see [LANL Earthquake Prediction Data Set](https://www.kaggle.com/c/LANL-Earthquake-Prediction/data)

<details><summary>CLICK TO EXPAND</summary>
<p>
  
```python

```

</p>
</details>


## III. Principal Components Analysis - PCA

## IV. Linear and Polynomial Regression 
#### Transform data
<details><summary>CLICK TO EXPAND</summary>
<p>
  
```python
target = pd.read_csv("extract_label_Jul08.csv", delimiter = ',')
target = target.as_matrix()
target = target[:,1]

features = pd.read_csv("extract_train_Jul08.csv", delimiter = ',')
features = features.as_matrix()
features = features[:, 1:17]
```

</p>
</details>

#### Help methods
##### Kfold Cross Validation for Linear and Polynomial Regression
<details><summary>CLICK TO EXPAND</summary>
<p>
  
```python
def K_Fold(features, target, numfolds, classifier):

    kf = KFold(n_splits=numfolds)
    kf.get_n_splits(features)

    i = 0
    mae = np.zeros(numfolds-1)
    coef = np.zeros([numfolds-1, 16])
    for train_index, test_index in kf.split(features):
        features_train, features_test = features[train_index], features[test_index]
        target_train, target_test = target[train_index], target[test_index]
    
        poly = PolynomialFeatures(degree=2)
        features_poly_train = features_train 
        features_poly_test = features_test
        if classifier == "polynomial" :
            features_poly_train = poly.fit_transform(features_train)
            features_poly_test = poly.fit_transform(features_test)
        elif classifier == "linear":
            features_poly_train = features_train 
            features_poly_test = features_test
        
        reg = LinearRegression().fit(features_poly_train, target_train)
        if classifier == "linear":
            if (i < numfolds - 1): 
                coef[i, :] = reg.coef_.reshape(1, 16)
        i = i+1
        if (i < numfolds):
            mae[i-1] = mean_absolute_error(target_test, reg.predict(features_poly_test))
            
    avrmae = (sum(mae)/(numfolds-1))
    var = (statistics.variance(mae))
    return avrmae, var, coef
```

</p>
</details>

##### Gradient mean and variance extraction method

<details><summary>CLICK TO EXPAND</summary>
<p>

```python
def mv(coefmat):
    mean = np.zeros(coefmat.shape[1])
    var = np.zeros(coefmat.shape[1])
    for i in range(coefmat.shape[1]):
        mean[i] = np.mean(coefmat[:, i])
        var[i] = np.std(coefmat[:, i])
    return mean, var
```

</p>
</details>

### Linear Regression

#### Perform linear regression on the data

```python
reg = LinearRegression().fit(features, target)

print("The loss values is: ", mean_absolute_error(target, reg.predict(features)))
```
The loss values is:  2.110853811043013

#### Perfom Kfold cross validation on the data

```python
i = np.array([5,10,50,100, 150])
mae = np.zeros(i.shape[0])
var = np.zeros(i.shape[0])
for numfold in range(i.shape[0]):
    m, v = K_Fold(features,target, i[numfold], "linear")
    mae[numfold] = m
    var[numfold] = v
plt.plot(i, mae, color = 'blue', label = 'Average MAE')
plt.plot(i, var, color = 'red', label = 'Variance of MAE')
plt.legend(loc='lower right')
```
![Linear Regression K Kold](https://github.com/hoangtung167/cx4240/blob/master/Graphs/Linear%20Regression%20K%20Fold.png)


#### Compare actual and predicted values of the outcome

```python
indx = range(target.shape[0])
plt.axis([0, target.shape[0], -0.1, 16])
plt.title("Comparison between predicted and actual target values")
plt.ylabel("Time before failure(s)")
plt.xlabel("Index")
plt.plot(indx, reg.predict(features), linewidth = 3, label = 'Pred')
plt.plot(indx, target, linewidth = 2, label = 'Actual')
plt.legend(loc='upper right')
plt.savefig('Linear Regression.png', dpi = 324)
```


![Linear Regression](https://github.com/hoangtung167/cx4240/blob/master/Graphs/Linear%20Regression.png)

#### Feature Importance

<details><summary>CLICK TO EXPAND</summary>
<p>
  
```python
fl = ['index', 'mean', 'std', 'skew', 'FFT_mean_real', 'FFT_mean_imag', 
     'FFT_std_real', 'FFT_std_max', 'Roll_std_p05', 'Roll_std_p30', 
      'Roll_std_p60', 'Roll_std_absDiff', 'Roll_mean_p05', 
      'Roll_mean_absDiff', 'MFCC_mean02', 'MFCC_mean16']
m, v, c = K_Fold(features,target, 100, "linear")
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
plt.savefig('bar_plot_with_error_bars.png', dpi = 324)
plt.show()
```

</p>
</details>

![Feature Importance](https://github.com/hoangtung167/cx4240/blob/master/Graphs/bar_plot_with_error_bars.png)

### Polynomial Regression
#### Perform polynomial regression on the data
 
```python
poly = PolynomialFeatures(degree=2)
features_poly = poly.fit_transform(features)
reg = LinearRegression().fit(features_poly, target)

print("The loss values is: ", mean_absolute_error(target, reg.predict(features_poly)))
```
The loss values is:  1.985654086901071

#### Perfom Kfold cross validation on the data

```python
i = np.array([5,10,50,100, 150])
mae = np.zeros(i.shape[0])
var = np.zeros(i.shape[0])
for numfold in range(i.shape[0]):
    m, v = K_Fold(features,target, i[numfold], "linear")
    mae[numfold] = m
    var[numfold] = v
plt.plot(i, mae, color = 'blue', label = 'Average MAE')
plt.plot(i, var, color = 'red', label = 'Variance of MAE')
plt.legend(loc='lower right')
```
![Polynomial Regression K Fold](https://github.com/hoangtung167/cx4240/blob/master/Graphs/Polynomial%20Regression%20K%20Fold.png)


#### Compare actual and predicted values of the outcome

```python
indx = range(target.shape[0])
plt.axis([0, target.shape[0], -0.1, 16])
plt.title("Comparison between predicted and actual target values")
plt.ylabel("Target Values")
plt.xlabel("Trial Number")
plt.plot(indx, reg.predict(features_poly), linewidth = 3)
plt.plot(indx, target, linewidth = 2)
```


![Polynomial Regression](https://github.com/hoangtung167/cx4240/blob/master/Graphs/Polynomial%20Regression.png)

### Comparision between Linear and Polynomial Regression


## V. Decision Tree/ Random Forest / LGB Classifier

## VI. Deep Learning/ Neural Nets

## VII. Summary

#### Compare the loss values and variances across methods

#### Compare the feature importance across methods
