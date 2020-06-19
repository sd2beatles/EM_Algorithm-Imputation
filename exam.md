```python
from GMM import EMImpute

data=np.array([[1.23,          3.2, np.nan,np.nan],
       [ -4.13801604,          4,          np.nan,2.23],
       [ -6.96117925,  -4.78845229,   1.21653198,1.2],
       [  6.57201047,   6.0520226 ,   8.87408451,1.3],
       [         np.nan,          np.nan,   9.17663177,2.2],
       [         np.nan,1.23,          np.nan,np.nan],
       [    3.2345,  np.nan,   4.44208904,2.24],
       [ -4.17891721,  np.nan,   4.29849034,np.nan],
       [-11.29647092,   1.28621915,          np.nan,np.nan],
       [12,11,3,np.nan]])
       
       
print(f'Data Before Imputation:\n{data}')
EMImpute(data,max_iter=100,k=3).fit_transform()
print('-'*100)
print(f'Data Before Imputation:\n{data}')
```
![image](https://user-images.githubusercontent.com/53164959/85133814-60509400-b276-11ea-8d0d-fde2fca6c012.png)


