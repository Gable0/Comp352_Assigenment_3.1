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
plt.show()

