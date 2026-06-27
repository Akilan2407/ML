import pandas as p
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
data = p.read_csv('Emp.csv')
print(data.head())
numerical = data.select_dtypes(include=['int64', 'float64']).columns
categorical = data.select_dtypes(include=['object', 'category']).columns
print("Numerical Attributes:")
print(list(numerical))
print("\nCategorical Attributes:")
print(list(categorical))
# Label Encoding
l = LabelEncoder()
data['Gender'] = l.fit_transform(data['Gender'])
print("\nGender Encoding:")
for i, category in enumerate(l.classes_):
    print(f"{category} -> {i}")
data['Education'] = l.fit_transform(data['Education'])
print("\nEducation Encoding:")
for i, category in enumerate(l.classes_):
    print(f"{category} -> {i}")
print("\nEncoded Dataset:")
print(data[['Gender', 'Education']].head())
# One Hot Encoding
data = p.get_dummies(
    data,
    columns=['Department', 'City', 'Work_Mode', 'Job_Role'],
    dtype=int
)
print("\nDataset After One-Hot Encoding:")
print(data.head())
# Store Salary before scaling
salary_before = data['Salary']
# Standard Scaling
scaler = StandardScaler()
data[['Age', 'Experience', 'Salary', 'Performance_Score']] = scaler.fit_transform(
    data[['Age', 'Experience', 'Salary', 'Performance_Score']]
)
# Store Salary after scaling
salary_after = data['Salary']
print("\nDataset After Standard Scaling:")
print(data[['Age', 'Experience', 'Salary', 'Performance_Score']].head())
# Min-Max Scaling
s = MinMaxScaler()
data[['Age', 'Experience', 'Salary', 'Performance_Score']] = s.fit_transform(
    data[['Age', 'Experience', 'Salary', 'Performance_Score']]
)
print("\nDataset After Min-Max Scaling:")
print(data[['Age', 'Experience', 'Salary', 'Performance_Score']].head())

# Visualization
plt.figure(figsize=(8,5))
plt.hist(salary_before, bins=10, edgecolor='black')
plt.title('Histogram of Salary Before Standard Scaling')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(8,5))
plt.hist(salary_after, bins=10, edgecolor='black')
plt.title('Histogram of Salary After Standard Scaling')
plt.xlabel('Scaled Salary')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(6,5))
plt.boxplot(salary_before)
plt.title('Boxplot of Salary Before Standard Scaling')
plt.ylabel('Salary')
plt.show()
plt.figure(figsize=(6,5))
plt.boxplot(salary_after)
plt.title('Boxplot of Salary After Standard Scaling')
plt.ylabel('Scaled Salary')
plt.show()
#comparision 
comparison = p.concat(
    [data[['Department', 'Department_Label']],
     department_onehot],
    axis=1
)
print("\nComparison of Label Encoding and One-Hot Encoding:")
print(comparison.head(10))