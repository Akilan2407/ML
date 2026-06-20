import pandas as pd
import numpy as np
data = pd.read_csv("Cafe.csv")
print(data.head())
missing_before = data.isnull().sum().sum()
data.replace(["ERROR", "UNKNOWN"], np.nan, inplace=True)
# Convert numeric columns
numeric_cols = ["Quantity", "Price Per Unit", "Total Spent"]
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")
print("Missing values before filling:")
print(data.isnull().sum())
# Fill missing values
data["Item"] = data["Item"].fillna("Unknown")
data["Quantity"] = data["Quantity"].fillna(data["Quantity"].mean())
data["Price Per Unit"] = data["Price Per Unit"].fillna(data["Price Per Unit"].mean())
data["Total Spent"] = data["Total Spent"].fillna(data["Total Spent"].mean())
data["Payment Method"] = data["Payment Method"].fillna("Unknown")
data["Location"] = data["Location"].fillna("Unknown")
data["Transaction Date"] = data["Transaction Date"].fillna("Unknown")
missing_after = data.isnull().sum().sum()
missing_removed = missing_before - missing_after
print("\nAfter filling missing values:")
print(data.isnull().sum())
# duplicates
print("\nBefore removing duplicates:")
print(data.shape[0])
duplicates_removed = data.duplicated().sum(
data.drop_duplicates(inplace=True)
print("After removing duplicates:")
print(data.shape[0])
#zero and negative values
invalid_records = data[
    (data["Quantity"] <= 0) |
    (data["Price Per Unit"] <= 0)
]
invalid_removed = len(invalid_records)
print("\nInvalid Records:")
print(invalid_records)
data = data[
    (data["Quantity"] > 0) &
    (data["Price Per Unit"] > 0)
]
data.reset_index(drop=True, inplace=True)
# date change
data["Transaction Date"] = pd.to_datetime(
    data["Transaction Date"],
    errors="coerce"
)
print("\nTransaction Date:")
print(data["Transaction Date"].head())
# standardizing text columns
data["Item"] = (
    data["Item"].fillna("Unknown").str.strip() .str.replace(r"\s+", " ", regex=True).str.title()
)
data["Payment Method"] = (
    data["Payment Method"].fillna("Unknown").str.strip().str.replace(r"\s+", " ", regex=True).str.title()
)
data["Location"] = (
    data["Location"].fillna("Unknown").str.strip().str.replace(r"\s+", " ", regex=True).str.title()
)
#to csv
data.to_csv("Cafe_Cleaned.csv", index=False)
#summary
print("\n========== DATA CLEANING SUMMARY ==========")
print("Missing values removed      :", missing_removed)
print("Duplicate records removed   :", duplicates_removed)
print("Invalid transactions removed:", invalid_removed)
print("Final number of records     :", len(data))