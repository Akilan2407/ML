import pandas as p
import os
def writeexcel(data, file_name, sheet_name):
     data.to_excel(file_name, sheet_name=sheet_name, index=False)    
def writecsv(data, file_name):
    data.to_csv(file_name, index=False)
#read data
data=p.read_csv("Emp.csv")
print(data.to_string())
#check for null
print(data.isnull().sum())
#check for duplicates
print(data.duplicated())
#data standard=> removing extra space
data['Department'] = data['Department'].str.strip().str.title()
data['Gender']=data['Gender'].str.strip().str.title()
print(data['Gender'])
print(data['Department'])
#print(data.head(20))
#outliers
Q1 = data['MonthlyIncome'].quantile(0.25)
Q3 = data['MonthlyIncome'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
outliers = data[(data['MonthlyIncome'] < lower_bound) | (data['MonthlyIncome'] > upper_bound)]
print("Outliers detected:\n", outliers[['EmployeeNumber','MonthlyIncome']])
data_clean = data[(data['MonthlyIncome'] >= lower_bound) & (data['MonthlyIncome'] <= upper_bound)]
print("Cleaned Data:\n", data_clean[['EmployeeNumber','MonthlyIncome']])
writeexcel(data,'output.xlsx','cleaned_data')
#output csv
writecsv(data,'output.csv')
print(data.info)
print(data.dtypes)

