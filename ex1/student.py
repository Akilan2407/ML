import pandas as p
import matplotlib.pyplot as pl
data=p.read_csv('student_data.csv')
#print(data.to_string())
#print first 10 rows
print(data.head(10))
#numbers of rows and columns in a dataset
rows, cols = data.shape
print(f"Number of rows: {rows}")
print(f"Number of columns: {cols}")
#attributes
print('Attributes in the dataset')
print(data.columns)
#missing data in the columns
print('NULL DATAS:')
print(data.isnull().sum())
#mean,median
cd = data["G1"].dropna()
mean = cd.mean()
print(f"mean of column 'G1' is '{mean}'")
median = cd.median()
print(f"median of column 'G1' is '{median}'")
#min and max
print(f"minimum of column 'G1' is {cd.min()}")
print(f"maximum of column 'G1' is {cd.max()}")
#numerical and categorical columns
numerical_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
print("Numerical:", numerical_cols)
print("Categorical:", categorical_cols)

#histogram
data['G1'].plot(kind='hist')
pl.show();
#correlation
print("\nCorrelation Matrix:")
print(data.corr(numeric_only=True))
#target variable
for col in data.columns:
    unique_vals = data[col].nunique()
    if unique_vals is 2:   
        print(f"Possible target: {col} (unique values: {unique_vals})")











