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


To access original data set, see [LANL Earthquake Prediction Data Set](https://www.kaggle.com/c/LANL-Earthquake-Prediction/data)

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
