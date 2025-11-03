import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore[reportMissingTypeStubs]
import warnings
warnings.filterwarnings('ignore')

#Problem 1: Weights.tsv
print("\n")
print("=====================================")
print("Problem 1: Weights.tsv")
print("=====================================")


#1.1 Import Data from weights.tsv to a Pandas Series
print("----------------------------------------------------")
print("1.1: Import Data from weights.tsv to a Pandas Series") 
print("----------------------------------------------------")
weights_df = pd.read_csv('weights.tsv', header=None) 
weights_lbs = pd.Series(weights_df[0].values)
print(f"Weight (lbs):\n{weights_lbs}")


#1.2 Create new series with weights in kilograms
print("\n")
print("----------------------------------------------------")
print("1.2: Create new series with weights in kilograms") 
print("----------------------------------------------------")
weights_kgs = (weights_lbs * 0.453592).round(2)
print(f"Weight (kgs):\n{weights_kgs}")


#1.3 Create new series with weights in kilograms
print("\n")
print("----------------------------------------------------")
print("1.3: Find the mean, median, and standard deviation")  
print("----------------------------------------------------")
print("Stuff for lbs:")
print(f"Mean (lbs): {weights_lbs.mean():.2f}lbs")
print(f"Median (lbs): {weights_lbs.median():.2f}lbs")
print(f"Standard Deviation (lbs): {weights_lbs.std():.2f}lbs")
print("\n")
print("Stuff for kgs:")
print(f"Mean (kgs): {weights_kgs.mean():.2f}kgs")
print(f"Median (kgs): {weights_kgs.median():.2f}kgs")
print(f"Standard Deviation (kgs): {weights_kgs.std():.2f}kgs")


#1.4 Plot a histogram of the weights in kilograms
print("\n")
print("----------------------------------------------------")
print("1.4: Plot a histogram of the weights in kilograms")  
print("----------------------------------------------------")
plt.figure(figsize=(10, 6))
plt.hist(weights_kgs, bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Weights in Kilograms', fontsize=16)
plt.xlabel('Weight (kgs)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('weights_histogram.png')
print("Histogram saved as 'weights_histogram.png'")
#plt.show()
plt.close()



#Problem 2: Boston.csv
print("\n")
print("=====================================")
print("Problem 2: Boston.csv")
print("=====================================")


#2.1 Import Data from Boston.csv
print("\n")
print("----------------------------------------------------")
print("2.1: Import Data from Boston.csv")  
print("----------------------------------------------------")
boston_df = pd.read_csv('boston.csv')
print(f"Data Dimentions: {boston_df.shape[0]} rows and {boston_df.shape[1]} columns")
print("First 5 rows of the dataset:")
print(boston_df.head())


#2.2 What is the MEDV (Owner-occupied home value) for the lowest NOX (Nitric oxides concentration) value?
print("\n")
print("----------------------------------------------------")
print("2.2: What is the MEDV for the lowest NOX value?")  
print("----------------------------------------------------")
min_nox_index = boston_df['NOX'].indexmin()
medv_at_min_nox = boston_df.loc[min_nox_index, 'MEDV']
min_nox = boston_df.loc[min_nox_index, 'NOX']
print(f"Lowest NOX concentration: {min_nox}")
print(f"MEDV at lowest NOX concentration: ${medv_at_min_nox}k") 


#2.3 Boxplot of CRIM and calculate IQR
print("\n")
print("----------------------------------------------------")
print("2.3: Boxplot of per capita crime rate (CRIM)")
print("----------------------------------------------------")
plt.figure(figsize=(10, 6))
plt.boxplot(boston_df['CRIM'], vert=False)
plt.title('Boxplot of Per Capita Crime Rate (CRIM)', fontsize=16)
plt.xlabel('CRIM', fontsize=14)
plt.grid(axis='x', alpha=0.75)
plt.savefig('crim_boxplot.png')
print("Boxplot saved as 'crim_boxplot.png'")    
#plt.show()
plt.close()
# Calculate IQR
Q1 = boston_df['CRIM'].quantile(0.25)
Q3 = boston_df['CRIM'].quantile(0.75)
IQR = Q3 - Q1
print(f"Q1 of CRIM: {Q1:.4f}")
print(f"Q3 of CRIM: {Q3:.4f}")
print(f"IQR of CRIM: {IQR:.4f}")


#2.4 Subset of Outliers and Compare AGE mean
print("\n")
print("----------------------------------------------------")
print("2.3: Boxplot of per capita crime rate (CRIM)")
print("----------------------------------------------------")
l_bound = Q1 - 1.5 * IQR
u_bound = Q3 + 1.5 * IQR
print(f"Outlier Bounds: Lower = {l_bound:.4f}, Upper = {u_bound:.4f}")
outliers_df = boston_df[(boston_df['CRIM'] < l_bound) | (boston_df['CRIM'] > u_bound)]
non_outliers_df = boston_df[(boston_df['CRIM'] >= l_bound) & (boston_df['CRIM'] <= u_bound)]

print(f"Number of Outliers: {len(outliers_df)}")
print(f"Number of Non-Outliers: {len(non_outliers_df)}")

#Compare Age means
mean_age_outliers = outliers_df['AGE'].mean()
mean_age_non_outliers = non_outliers_df['AGE'].mean()
total_mean_age = boston_df['AGE'].mean() 

print("\nMean AGE Comparison:")
print(f"Mean AGE of Outliers: {mean_age_outliers:.2f}")
print(f"Mean AGE of Non-Outliers: {mean_age_non_outliers:.2f}")
print(f"Overall Mean AGE: {total_mean_age:.2f}")

print(f"\nObservation: The mean AGE of outlier neighborhoods is {'higher' if mean_age_outliers > mean_age_non_outliers else 'lower'} ")
print("than that of non-outlier neighborhoods. This suggests that neighborhoods with extreme crime rates tend to have older housing ")
print("stock compared to those with typical crime rates.")


#2.5 Scatter plot of DIS vs NOX with Correlation
print("\n")
print("----------------------------------------------------")
print("2.5: Scatter plot of DIS vs NOX with Correlation")
print("----------------------------------------------------")
plt.figure(figsize=(10, 6))
plt.scatter(boston_df['DIS'], boston_df['NOX'], color='purple', alpha=0.6)
plt.title('Scatter Plot of DIS vs NOX', fontsize=16)
plt.xlabel('DIS (Weighted distances to five Boston employment centres)', fontsize=14)
plt.ylabel('NOX (Nitric oxides concentration)', fontsize=14)
plt.grid(alpha=0.75)
plt.savefig('dis_vs_nox_scatter.png')
print("Scatter plot saved as 'dis_vs_nox_scatter.png'")
#plt.show()
plt.close()

correlation_DISvsNOX = boston_df['DIS'].corr(boston_df['NOX'])
print(f"\nCorrelation coefficient between DIS and NOX: {correlation_DISvsNOX:.4f}") 
print(f"Observation: There is a strong negative correlation between DIS and NOX, indicating")
print("that as the distance to employment centers increases, the nitric oxides concentration tends to decrease.")
print("This suggests that areas further from employment centers may have lower pollution levels.")


#2.6 Scatter plot of RAD vs TAX with Correlation
print("\n")
print("----------------------------------------------------")
print("2.6: Scatter plot of DIS vs NOX with Correlation")
print("----------------------------------------------------")
plt.figure(figsize=(10, 6))
plt.scatter(boston_df['RAD'], boston_df['TAX'], color='green', alpha=0.6)
plt.title('Scatter Plot of RAD vs TAX', fontsize=16)
plt.xlabel('RAD (Index of accessibility to radial highways)', fontsize=14)
plt.ylabel('TAX (Full-value property-tax rate per $10,000)', fontsize=14)
plt.grid(alpha=0.75)
plt.savefig('rad_vs_tax_scatter.png')
print("Scatter plot saved as 'rad_vs_tax_scatter.png'")
#plt.show()
plt.close()

correlation_RADvsTAX = boston_df['RAD'].corr(boston_df['TAX'])
print(f"\nCorrelation coefficient between RAD and TAX: {correlation_RADvsTAX:.4f}") 
print(f"Observation: There is a strong correlation between RAD and TAX, indicating that as accessibility to radial highways increases, ") 
print("the property-tax rate also tends to increase. This suggests that properties with better highway access may be valued higher, ")
print("leading to higher taxes.") 

#Check discrete RAD
print("\n")
print(f"Unique values in RAD: {sorted(boston_df['RAD'].unique())}")
print("Observation: The RAD variable is discrete, representing specific indices of accessibility to radial highways.")

