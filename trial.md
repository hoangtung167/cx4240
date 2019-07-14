## III. Principal Components Analysis - PCA
<details><summary>CLICK TO EXPAND</summary>
<p>
 
 ### Importing Required Packages
<details><summary>CLICK TO EXPAND</summary>
<p>

```python
  import numpy as np
  import seaborn as sns
  import matplotlib.pyplot as plt
  import pandas as pd
  from sklearn.preprocessing import StandardScaler
  from sklearn.decomposition import PCA
  ```
 </p>
 </details>
 ### Data Upload 
<details><summary>CLICK TO EXPAND</summary>
<p>

```python
train = pd.read_csv('extract_train_Jul08.csv')
train = train.drop(['index'], axis = 1)
train = train.drop(train.columns[0],axis = 1)
```
</p>
</details>
 ### Standardize Data for PCA input

<details><summary>CLICK TO EXPAND</summary>
<p>
 
```python
scaler=StandardScaler() #instantiate
scaler.fit(train) # compute the mean and standard which will be used in the next command
X_scaled=scaler.transform(train)
```
</p>
</details>
 ### Fitting the PCA (16 principal components)
<details><summary>CLICK TO EXPAND</summary>
<p>
 
```python
pca=PCA() 
pca.fit(X_scaled) 
X_pca=pca.transform(X_scaled)
```

```python
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
```

```python
ex_variance=np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio)
```
</p>
</details>
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

</p>
</details>
