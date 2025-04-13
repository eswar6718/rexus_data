import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data_gov_bldg_rexus.csv")

df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('\r', '').str.replace(' ', '_')

column_renames = {
    'Construction_Date': 'Year_Built',
    'Bldg_ANSI_Usable': 'Square_Footage',
    'Agency_Name': 'Agency',
    'Bldg_State': 'State',
    'Bldg_City': 'City',
    'Property_Type': 'Property_Type'
}
df.rename(columns={k: v for k, v in column_renames.items() if k in df.columns}, inplace=True)

if 'Year_Built' in df.columns:
    df['Year_Built'] = df['Year_Built'].astype(str).str.extract(r'(\d{4})').astype(float)

if 'Square_Footage' in df.columns:
    df['Square_Footage'] = pd.to_numeric(df['Square_Footage'], errors='coerce')

if 'Year_Built' in df.columns:
    df['Year_Built'] = df['Year_Built'].fillna(df['Year_Built'].median())
if 'Square_Footage' in df.columns:
    df['Square_Footage'] = df['Square_Footage'].fillna(df['Square_Footage'].median())
for col in ['Property_Type', 'Agency', 'State', 'City']:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

columns_to_drop = ['ABA_Accessibility_Flag', 'Historical_Type', 'Owned_Leased']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

df.to_csv("cleaned_rexus_dataset.csv", index=False)

print("\n Cleaned Data Preview:")
print(df.head())

sns.set(style="whitegrid")

# Bar chart: Count of buildings by State
plt.figure(figsize=(12, 6))
state_counts = df['State'].value_counts()
state_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Buildings by State')
plt.xlabel('State')
plt.ylabel('Number of Buildings')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Pie chart: Property Type Distribution (Basic version)
plt.figure(figsize=(8, 8))
property_counts = df['Property_Type'].value_counts()
plt.pie(property_counts.values, labels=property_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Property Types')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Box plot: Square Footage distribution by Property Type
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Property_Type', y='Square_Footage', showfliers=False)
plt.title('Square Footage Distribution by Property Type (Box Plot)')
plt.xlabel('Property Type')
plt.ylabel('Square Footage')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Correlation Heatmap
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()

# KDE Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(df['Square_Footage'], fill=True, color='green')
plt.title('KDE Plot: Distribution of Square Footage')
plt.xlabel('Square Footage')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

# IQR Calculation for Outlier Detection
Q1 = df['Square_Footage'].quantile(0.25)
Q3 = df['Square_Footage'].quantile(0.75)
IQR = Q3 - Q1

# Determine the lower and upper bounds for Square Footage
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\n Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

# Outlier Detection and Removal based on IQR
outliers = df[(df['Square_Footage'] < lower_bound) | (df['Square_Footage'] > upper_bound)]

print("\n Detected Outliers:")
print(outliers)

# Removing outliers
df_cleaned = df[(df['Square_Footage'] >= lower_bound) & (df['Square_Footage'] <= upper_bound)]

# Scatter Plot BEFORE Outlier Removal
plt.figure(figsize=(10, 6))
plt.scatter(df['Year_Built'], df['Square_Footage'], alpha=0.4, color='skyblue')
plt.title('Scatter Plot (Before Outlier Removal): Year Built vs Square Footage')
plt.xlabel('Year Built')
plt.ylabel('Square Footage')
plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter Plot AFTER Outlier Removal
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['Year_Built'], df_cleaned['Square_Footage'], alpha=0.4, color='orange')
plt.title('Scatter Plot (After Outlier Removal): Year Built vs Square Footage')
plt.xlabel('Year Built')
plt.ylabel('Square Footage')
plt.grid(True)
plt.tight_layout()
plt.show()

#Histogram for Square Footage distribution
plt.figure(figsize=(10, 6))
plt.hist(df['Square_Footage'].dropna(), bins=30, color='teal', edgecolor='black')
plt.title('Histogram of Square Footage (Building Sizes)')
plt.xlabel('Square Footage')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
