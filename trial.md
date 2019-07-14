## III. Principal Components Analysis - PCA

Principal component analysis (PCA) is a technique used for understanding the dimensional structure of a data set. PCA transforms data in a way that converts a set of orthogonally correlated observations into a set of linearly uncorrelated variables called principal components.  This transformation maximizes the largest possible variances for each principal component. There can be as many principle components are there are feature dimensions in the data set. Each principal component accounts for the largest possible variance between entries. 

In this work we use three different visualization methods to help understand the dimensional structure of the data and reduce the dimensionality of the dataset. 

 ### Importing Required Packages and Files
<details><summary>CLICK TO EXPAND</summary>
<p>

```python
  import numpy as np
  import seaborn as sns
  import matplotlib.pyplot as plt
  import pandas as pd
  from sklearn.preprocessing import StandardScaler
  from sklearn.decomposition import PCA

train = pd.read_csv('extract_train_Jul08.csv')
train = train.drop(['index'], axis = 1)
train = train.drop(train.columns[0],axis = 1)
```
</p>
</details>
 
 ### Running PCA Data
<details><summary>CLICK TO EXPAND</summary>
<p>
 
```python
scaler=StandardScaler() #instantiate
scaler.fit(train) # compute the mean and standard which will be used in the next command
X_scaled=scaler.transform(train)
pca=PCA() 
pca.fit(X_scaled) 
X_pca=pca.transform(X_scaled)
ex_variance=np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio)
```
</p>
</details>

 ### Pricipal Component Proportioanlity
<details><summary>CLICK TO EXPAND</summary>
<p>

```python
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
```
</p>
</details>

The x-axis of the graph below labels each principal component for the featurized data set, while the y-axis accounts for the proportionality of the total variance contained within the data set. As expected, the first principal component accounts for the largest amount of variance. Each consecutive principal component accounts for more variance than the one after it. 

The red line shows the cumulative proportional variance after each principal component is formed. The dashed line is an indication of 99% variance of the data. One can see that the dashed line crosses the cumulative sum (red) line at the 9th principal component. This indicated that 99% of the variance within the data is accounted for when the dimensionality of the data is reduced from 16 dimensions down to 9 dimensions. 

![Principal Components Visualization](https://github.com/hoangtung167/cx4240/blob/master/Graphs/principal_component_visualization.png)

 ### Feature Variance for Pricipal Component 1 & 2
<details><summary>CLICK TO EXPAND</summary>
<p>
 
```python
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
```
</p>
</details>
The two plots show the contributing variance of each feature in the first and second principal component. Yellow indicates a high positive variance while purple indicates a high negative variance. In the first principal component the features contributing to the most variance are the ‘Roll_std_pXX’ features as well as the “MFCC_mean02” components. In the second principal component the “mean”, “FFT_std_max”, and “index” features contribute to the most variance. Knowing this correlation relationship could provide a framework for identifying the most important features within the model. 

![First Principal Component](https://github.com/hoangtung167/cx4240/blob/master/Graphs/first_principal_component.png)

![second Principal Component](https://github.com/hoangtung167/cx4240/blob/master/Graphs/second_principal_component.png)

 ### Visualizing Feature Correlation
<details><summary>CLICK TO EXPAND</summary>
<p>
 
```python
features = test.columns
plt.figure(figsize=(8,8))
s=sns.heatmap(test.corr(),cmap='coolwarm') 
s.set_yticklabels(s.get_yticklabels(),rotation=30,fontsize=7)
s.set_xticklabels(s.get_xticklabels(),rotation=30,fontsize=7)
plt.show()
```
</p>
</details>

The final graph within this section is a heat map which shows the correlation between different features. Dark red indicates that features have a strong positive correlation while dark blue indicates that there is a strong negative correlation. This heat map provides further insight into which features are linearly independent and which variables linearly dependent. For example, the “Roll_std_p60” and “skew” features are linearly independent and have nearly zero correlation between each feature. On the other hand, “Roll_std_60” is correlated strongly with 7 features. 

![Feature Correlation](https://github.com/hoangtung167/cx4240/blob/master/Graphs/heat_map.png)

 ### Saving Reduced Dimensionality Matrix and Feature Importance
<details><summary>CLICK TO EXPAND</summary>
<p>
 
```python
a = np.abs(pca.components_[0])
a = a/np.max(a)
df = pd.DataFrame()
df['features'] = test.columns
df['importance'] = a
df.to_csv('PCA_extracted.csv')
print(df.shape)

pca=PCA(n_components = 9) 
pca.fit(X_scaled) 
X_pca=pca.transform(X_scaled)
df = pd.DataFrame(X_pca)
df.to_csv('pca_exported_9features.csv')
```
</p>
</details>

