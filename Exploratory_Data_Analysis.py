import csv
import math
import statistics
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pd = None

try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    sns = None

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    plt = None


BASE_DIR = Path(__file__).resolve().parent


def read_weights(path: Path) -> list[float]:
    weights = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                weights.append(float(value))
    return weights


def read_boston(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        headers = [column.strip() for column in next(reader)]
        rows = []
        for raw_row in reader:
            if not raw_row:
                continue
            cleaned_row: dict[str, float] = {}
            for header, raw_value in zip(headers, raw_row):
                value = raw_value.strip().strip('"').replace("\t", "")
                cleaned_row[header] = float(value)
            rows.append(cleaned_row)
    return headers, rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile() arg is an empty sequence")
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[int(position)]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * (position - lower_index)


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) != len(y_values):
        raise ValueError("Sequences must be the same length for correlation.")
    if len(x_values) < 2:
        return float("nan")
    mean_x = statistics.fmean(x_values)
    mean_y = statistics.fmean(y_values)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in x_values) * sum((y - mean_y) ** 2 for y in y_values)
    )
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def column(rows: list[dict[str, float]], name: str) -> list[float]:
    return [row[name] for row in rows]


# Problem 1: Weights.tsv
print("\n")
print("=====================================")
print("Problem 1: Weights.tsv")
print("=====================================")

# 1.1 Import Data from weights.tsv to a sequence
print("----------------------------------------------------")
print("1.1: Import Data from weights.tsv to a sequence")
print("----------------------------------------------------")
weights_lbs = read_weights(BASE_DIR / "weights.tsv")
print("Weight (lbs):")
for weight in weights_lbs:
    print(weight)

# 1.2 Create new sequence with weights in kilograms
print("\n")
print("----------------------------------------------------")
print("1.2: Create new series with weights in kilograms")
print("----------------------------------------------------")
weights_kgs = [round(weight * 0.453592, 2) for weight in weights_lbs]
print("Weight (kgs):")
for weight in weights_kgs:
    print(weight)

# 1.3 Find the mean, median, and standard deviation
print("\n")
print("----------------------------------------------------")
print("1.3: Find the mean, median, and standard deviation")
print("----------------------------------------------------")
print("Stuff for lbs:")
print(f"Mean (lbs): {statistics.fmean(weights_lbs):.2f}lbs")
print(f"Median (lbs): {statistics.median(weights_lbs):.2f}lbs")
print(f"Standard Deviation (lbs): {statistics.stdev(weights_lbs):.2f}lbs")
print("\n")
print("Stuff for kgs:")
print(f"Mean (kgs): {statistics.fmean(weights_kgs):.2f}kgs")
print(f"Median (kgs): {statistics.median(weights_kgs):.2f}kgs")
print(f"Standard Deviation (kgs): {statistics.stdev(weights_kgs):.2f}kgs")

# 1.4 Plot a histogram of the weights in kilograms
print("\n")
print("----------------------------------------------------")
print("1.4: Plot a histogram of the weights in kilograms")
print("----------------------------------------------------")
if plt is None:
    print("Matplotlib is not available; skipping histogram plot.")
else:
    plt.figure(figsize=(10, 6))
    plt.hist(weights_kgs, bins=10, color="skyblue", edgecolor="black")
    plt.title("Histogram of Weights in Kilograms", fontsize=16)
    plt.xlabel("Weight (kgs)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis="y", alpha=0.75)
    plt.savefig(BASE_DIR / "weights_histogram.png")
    print("Histogram saved as 'weights_histogram.png'")
    plt.close()

# Problem 2: Boston.csv
print("\n")
print("=====================================")
print("Problem 2: Boston.csv")
print("=====================================")

# 2.1 Import Data from Boston.csv
print("\n")
print("----------------------------------------------------")
print("2.1: Import Data from Boston.csv")
print("----------------------------------------------------")
boston_headers, boston_rows = read_boston(BASE_DIR / "boston.csv")
print(f"Data Dimensions: {len(boston_rows)} rows and {len(boston_headers)} columns")
print("First 5 rows of the dataset:")
for index, row in enumerate(boston_rows[:5], start=1):
    formatted_row = ", ".join(f"{header}: {row[header]}" for header in boston_headers)
    print(f"Row {index}: {formatted_row}")

# 2.2 What is the MEDV (Owner-occupied home value) for the lowest NOX (Nitric oxides concentration) value?
print("\n")
print("----------------------------------------------------")
print("2.2: What is the MEDV for the lowest NOX value?")
print("----------------------------------------------------")
min_nox_row = min(boston_rows, key=lambda item: item["NOX"])
print(f"Lowest NOX concentration: {min_nox_row['NOX']}")
print(f"MEDV at lowest NOX concentration: ${min_nox_row['MEDV']}k")

# 2.3 Boxplot of CRIM and calculate IQR
print("\n")
print("----------------------------------------------------")
print("2.3: Boxplot of per capita crime rate (CRIM)")
print("----------------------------------------------------")
crim_values = column(boston_rows, "CRIM")
if plt is None:
    print("Matplotlib is not available; skipping CRIM boxplot.")
else:
    plt.figure(figsize=(10, 6))
    plt.boxplot(crim_values, vert=False)
    plt.title("Boxplot of Per Capita Crime Rate (CRIM)", fontsize=16)
    plt.xlabel("CRIM", fontsize=14)
    plt.grid(axis="x", alpha=0.75)
    plt.savefig(BASE_DIR / "crim_boxplot.png")
    print("Boxplot saved as 'crim_boxplot.png'")
    plt.close()
q1_crim = percentile(crim_values, 0.25)
q3_crim = percentile(crim_values, 0.75)
iqr_crim = q3_crim - q1_crim
print(f"Q1 of CRIM: {q1_crim:.4f}")
print(f"Q3 of CRIM: {q3_crim:.4f}")
print(f"IQR of CRIM: {iqr_crim:.4f}")

# 2.4 Subset of Outliers and Compare AGE mean
print("\n")
print("----------------------------------------------------")
print("2.4: Subset outlier crime rates and compare AGE means")
print("----------------------------------------------------")
lower_bound = q1_crim - 1.5 * iqr_crim
upper_bound = q3_crim + 1.5 * iqr_crim
print(f"Outlier Bounds: Lower = {lower_bound:.4f}, Upper = {upper_bound:.4f}")
outliers = [row for row in boston_rows if row["CRIM"] < lower_bound or row["CRIM"] > upper_bound]
non_outliers = [row for row in boston_rows if lower_bound <= row["CRIM"] <= upper_bound]
print(f"Number of Outliers: {len(outliers)}")
print(f"Number of Non-Outliers: {len(non_outliers)}")

mean_age_outliers = statistics.fmean(column(outliers, "AGE")) if outliers else float("nan")
mean_age_non_outliers = statistics.fmean(column(non_outliers, "AGE")) if non_outliers else float("nan")
mean_age_total = statistics.fmean(column(boston_rows, "AGE"))

print("\nMean AGE Comparison:")
print(f"Mean AGE of Outliers: {mean_age_outliers:.2f}")
print(f"Mean AGE of Non-Outliers: {mean_age_non_outliers:.2f}")
print(f"Overall Mean AGE: {mean_age_total:.2f}")

comparison = "higher" if mean_age_outliers > mean_age_non_outliers else "lower"
print(f"\nObservation: The mean AGE of outlier neighborhoods is {comparison} ")
print("than that of non-outlier neighborhoods. This suggests that neighborhoods with extreme crime rates tend to have older housing ")
print("stock compared to those with typical crime rates.")

# 2.5 Scatter plot of DIS vs NOX with Correlation
print("\n")
print("----------------------------------------------------")
print("2.5: Scatter plot of DIS vs NOX with Correlation")
print("----------------------------------------------------")
dis_values = column(boston_rows, "DIS")
nox_values = column(boston_rows, "NOX")
if plt is None:
    print("Matplotlib is not available; skipping DIS vs NOX scatter plot.")
else:
    plt.figure(figsize=(10, 6))
    plt.scatter(dis_values, nox_values, color="purple", alpha=0.6)
    plt.title("Scatter Plot of DIS vs NOX", fontsize=16)
    plt.xlabel("DIS (Weighted distances to five Boston employment centres)", fontsize=14)
    plt.ylabel("NOX (Nitric oxides concentration)", fontsize=14)
    plt.grid(alpha=0.75)
    plt.savefig(BASE_DIR / "dis_vs_nox_scatter.png")
    print("Scatter plot saved as 'dis_vs_nox_scatter.png'")
    plt.close()

correlation_dis_nox = pearson_correlation(dis_values, nox_values)
print(f"\nCorrelation coefficient between DIS and NOX: {correlation_dis_nox:.4f}")
print("Observation: There is a strong negative correlation between DIS and NOX, indicating")
print("that as the distance to employment centers increases, the nitric oxides concentration tends to decrease.")
print("This suggests that areas further from employment centers may have lower pollution levels.")

# 2.6 Scatter plot of RAD vs TAX with Correlation
print("\n")
print("----------------------------------------------------")
print("2.6: Scatter plot of RAD vs TAX with Correlation")
print("----------------------------------------------------")
rad_values = column(boston_rows, "RAD")
tax_values = column(boston_rows, "TAX")
if plt is None:
    print("Matplotlib is not available; skipping RAD vs TAX scatter plot.")
else:
    plt.figure(figsize=(10, 6))
    plt.scatter(rad_values, tax_values, color="green", alpha=0.6)
    plt.title("Scatter Plot of RAD vs TAX", fontsize=16)
    plt.xlabel("RAD (Index of accessibility to radial highways)", fontsize=14)
    plt.ylabel("TAX (Full-value property-tax rate per $10,000)", fontsize=14)
    plt.grid(alpha=0.75)
    plt.savefig(BASE_DIR / "rad_vs_tax_scatter.png")
    print("Scatter plot saved as 'rad_vs_tax_scatter.png'")
    plt.close()

correlation_rad_tax = pearson_correlation(rad_values, tax_values)
print(f"\nCorrelation coefficient between RAD and TAX: {correlation_rad_tax:.4f}")
print("Observation: There is a strong correlation between RAD and TAX, indicating that as accessibility to radial highways increases, ")
print("the property-tax rate also tends to increase. This suggests that properties with better highway access may be valued higher, ")
print("leading to higher taxes.")

# Check discrete RAD
print("\n")
print(f"Unique values in RAD: {sorted({int(value) for value in rad_values})}")
print("Observation: The RAD variable is discrete, representing specific indices of accessibility to radial highways.")



#Problem 3: Tips from Seaborn Library
print("\n")
print("=====================================")
print("Problem 3: Tips from Seaborn Library")
print("=====================================")


#3.1 Import Tips dataset
print("\n")
print("----------------------------------------------------")
print("3.1: Import Tips dataset")
print("----------------------------------------------------")

#get the tips dataset from seaborn
tips_df = sns.load_dataset('tips')
tips_df.head()

print(f"Tips Dataset Dimensions: {tips_df.shape[0]} rows and {tips_df.shape[1]} columns")
print("First 5 rows of the dataset:")
print(tips_df.head())


#3.2 Days of the Week and Highest Bill 
print("\n")
print("----------------------------------------------------")
print("3.2: Days of the Week and Highest Bill")
print("----------------------------------------------------")
max_bill_index = tips_df['total_bill'].idxmax()
day_of_max_bill = tips_df.loc[max_bill_index, 'day']
max_bill_amount = tips_df.loc[max_bill_index, 'total_bill']
print(f"Day with Highest Total Bill: {day_of_max_bill}")
print(f"Highest Total Bill Amount: ${max_bill_amount:.2f}")


#3.3 Lunch vs Dinner Smoker Analysis
print("\n")
print("----------------------------------------------------")
print("3.3: Lunch vs Dinner Smoker Analysis")
print("----------------------------------------------------")

#Lunch vs Dinner total
time_count = tips_df.groupby('time').size().to_frame('Total Count')
print("Total Customers by Time:")
print(time_count)

#Lunch vs Dinner smokers
smoker_time_count = tips_df.groupby(['time', 'smoker']).size().unstack(fill_value=0)
print("\nSmokers by Time:")
print(smoker_time_count)

#Calculate Total Smokers vs Non-Smokers
smoker_total = tips_df('time')['smoker'].value_counts().unstack(fill_value=0)
total = smoker_total.sum(axis=1)
smokers_yesNo = smoker_total['Yes']

#Join Dataframe
# Join dataframes
combined_diners = pd.DataFrame({
    'total_count': time_count['count'],
    'smokers': smokers_yesNo
})

combined_diners['smoker_percentage'] = (combined_diners['smokers'] / combined_diners['total_count'] * 100).round(2)
print("\nSmoker Analysis by Time:")
print(combined_diners)

print("\nObservation: Dinner has a higher total number of customers as well as a higher number of smokers compared to lunch.")


#3.4 Boxplot of Total Bill by Sex
print("\n")
print("----------------------------------------------------")
print("3.4: Boxplot of Total Bill by Sex")
print("----------------------------------------------------")
plt.figure(figsize=(10, 6))
sns.boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='total_bill', data=tips_df, palette='Set2')
plt.title('Boxplot of Total Bill by Sex', fontsize=16)
plt.xlabel('Sex', fontsize=14)  
plt.ylabel('Total Bill ($)', fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('total_bill_by_sex_boxplot.png')
print("Boxplot saved as 'total_bill_by_sex_boxplot.png'")
#plt.show()
plt.close()     

#Outliers Count
male_tips = tips_df[tips_df['sex'] == 'Male']['tip']
female_tips = tips_df[tips_df['sex'] == 'Female']['tip']

def count_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    l_bound = Q1 - 1.5 * IQR
    u_bound = Q3 + 1.5 * IQR
    outliers = data[(data < l_bound) | (data > u_bound)]
    return len(outliers)

male_outliers_count = count_outliers(male_tips)
female_outliers_count = count_outliers(female_tips) 

#Analysis by Kawalski 
print(f"\nNumber of Outliers in Tips")
print(f"Males: {male_outliers_count}")
print(f"Females: {female_outliers_count}")

print("\nObservation: The boxplot indicates that males generally have a higher tip amount compared to females.")

#3.5 Boxblot of Tip Percentage by Sex that are below 70%
print("\n")
print("------------------------------------------------------------")
print("3.5: Boxplot of Tip Percentage by Sex that are below 70%")
print("------------------------------------------------------------")
tips_df_filtered = tips_df[tips_df['tip'] / tips_df['total_bill'] < 0.7]
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y=tips_df_filtered['tip'] / tips_df_filtered['total_bill'], data=tips_df_filtered, palette='Set3')
plt.title('Boxplot of Tip Percentage by Sex (Below 70%)', fontsize=16)
plt.xlabel('Sex', fontsize=14)  
plt.ylabel('Tip Percentage', fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('tip_percentage_by_sex_boxplot.png')
print("Boxplot saved as 'tip_percentage_by_sex_boxplot.png'")
#plt.show()
plt.close() 

male_percentages = tips_df_filtered[tips_df_filtered['sex'] == 'Male']['tip_percentage']
female_percentages = tips_df_filtered[tips_df_filtered['sex'] == 'Female']['tip_percentage']

#outliers count
male_percentage_outliers = count_outliers(male_percentages)
female_percentage_outliers = count_outliers(female_percentages)

#Skewness test
male_is_skewed = male_percentages.skew()
female_is_skewed = female_percentages.skew()

#Analysis by Analysis person 
print(f"\nOutlier/Skewness Analysis of Tip Percentages")
print(f"Male Outliers: {male_percentage_outliers}, Skewness: {male_is_skewed:.4f}")
print(f"Female Outliers: {female_percentage_outliers}, Skewness: {female_is_skewed:.4f}")
print(f"\n{'Males' if male_percentage_outliers > female_outliers_count else 'Females'} have more outliers in tip percentage.")
print(f"{'Males' if abs(male_is_skewed) < abs(female_is_skewed) else 'Females'} have less skewed distribution of data.")



#Problem 4: Avocado.csv
print("\n")
print("=====================================")
print("Problem 4: Avocado.csv")
print("=====================================")


#4.1 Import Avocado dataset
print("\n")
print("----------------------------------------------------")
print("4.1: Import Avocado dataset")
print("----------------------------------------------------")
print("\n")
avocado_df = pd.read_csv('avocado.csv')
print(f"Dataset Dimensions: {avocado_df.shape[0]} rows and {avocado_df.shape[1]} columns")
print(f"\nMissing Values in Each Column:\n{avocado_df.isnull().sum()}")

#Missing Values Analysis
missing_values = pd.DataFrame({
    'Missing Values': avocado_df.isnull().sum(),
    'Percentage (%)': (avocado_df.isnull().sum() / len(avocado_df) * 100).round(2)
})

print(f"\nMissing values summary:\n{missing_values[missing_values['Missing_Count'] > 0]}")

#Missing Values Handling
print("\nHandling missing values...")

print("Fill 'Type' missing values with mode...")
avocado_df['Type'].fillna(avocado_df['Type'].mode()[0])

print("Fill 'Year' missing values with median...")
avocado_df['Year'].fillna(avocado_df['Year'].median())['Year'].fillna(avocado_df['Year'].median())

print("Fill AllSize with 0 for missing values...")
avocado_df['AllSizes'] = avocado_df['AllSizes'].fillna(0) 

print(f"Total Missing Values: \n{avocado_df.isnull().sum()}") 

#4.2 Convert to Categorical Data Type
print("\n")
print("----------------------------------------------------")
print("4.2: Convert to Categorical Data Type")
print("----------------------------------------------------")
avocado_df['Type'] = avocado_df['Type'].astype('category')
avocado_df['Year'] = avocado_df['Year'].astype('category')
avocado_df['Region'] = avocado_df['Region'].astype('category')

#Exclude TotalUS and West regions from Region
avocado_df_no_west = avocado_df[~avocado_df['Region'].isin(['TotalUS', 'West'])]
avocado_df_no_west['Date'] = pd.to_datetime(avocado_df['Date'], format='%Y-%m-%d')
avocado_df_no_west = avocado_df_no_west.sort_values(by='Date')

print(f"Filtered Dataset Dimensions (excluding TotalUS and West): {avocado_df_no_west.shape[0]} rows and {avocado_df_no_west.shape[1]} columns")

#2016 vs 2017
print("\n")
mean_2016 = avocado_df_no_west[avocado_df_no_west['Year'] == 2016]['AveragePrice'].mean()
mean_2017 = avocado_df_no_west[avocado_df_no_west['Year'] == 2017]['AveragePrice'].mean()
print(f"Mean Average Price in 2016: ${mean_2016:.2f}")
print(f"Mean Average Price in 2017: ${mean_2017:.2f}")
print("Observation: The mean average price of avocados increased from 2016 to 2017, indicating a ")
print("rising trend in avocado prices during this period.")


#4.3 Total Volume by Region Bar Chart
print("\n")
print("----------------------------------------------------")
print("4.3: Total Volume by Region Bar Chart")
print("----------------------------------------------------")

vol_by_region = avocado_df_no_west.groupby('Region')['Total Volume'].sum().sort_values(ascending=True)
print(f"\nTotal Volume by Region:\n{vol_by_region}")

#Create Bar Chart
plt.figure(figsize=(12, 8))
vol_by_region.plot(kind='barh', color='teal')
plt.title('Total Volume of Avocados Sold by Region', fontsize=16)
plt.xlabel('Total Volume', fontsize=14)
plt.ylabel('Region', fontsize=14)
plt.grid(axis='x', alpha=0.75)
plt.savefig('total_volume_by_region_bar_chart.png')     
print("Bar chart saved as 'total_volume_by_region_bar_chart.png'")
#plt.show()
plt.close()

highest_volume = vol_by_region.idxmax()
print(f"\nRegion with Highest Total Volume: {highest_volume} with {vol_by_region.max():,.0f} units sold.")

#Subset thing with histogram
state_vol_data = avocado_df_no_west[avocado_df_no_west['Region'].str.contains('California|Texas|Florida')]
#create histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=state_vol_data, x='Total Volume', hue='Region', element='step', stat='density', common_norm=False, bins=30)
plt.title('Histogram of Total Volume for California, Texas, and Florida', fontsize=16)
plt.xlabel('Total Volume', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(alpha=0.75)
plt.savefig('state_total_volume_histogram.png')
print("Histogram saved as 'state_total_volume_histogram.png'")
#plt.show()
plt.close()

print("\nObservation: California has the highest total volume of avocados sold among the regions, ")
print("likely due to its large population and strong demand for avocados. ")
print(f"Mean Price: ${state_vol_data[state_vol_data['Region'] == 'California']['AveragePrice'].mean():.2f} | ")
print(f"Median Price: ${state_vol_data[state_vol_data['Region'] == 'California']['AveragePrice'].median():.2f}")
print(f"Price Range: ${state_vol_data[state_vol_data['Region'] == 'California']['AveragePrice'].max() - state_vol_data[state_vol_data['Region'] == 'California']['AveragePrice'].min():.2f}")

#Correlation between Average Price and Total Volume
correlation_price_volume = state_vol_data['AveragePrice'].corr(state_vol_data['Total Volume'])
print(f"\nCorrelation coefficient between Average Price and Total Volume: {correlation_price_volume:.4f}") 
print("Observation: There is a weak negative correlation between Average Price and Total Volume, suggesting that as the ")
print("price of avocados increases, the total volume sold tends to decrease slightly. This indicates that higher prices ")
print("may lead to reduced demand.")


#4.4 Timeline Plot
print("\n")
print("----------------------------------------------------")
print("4.4: Timeline Plot of Average Price Over Time")
print("----------------------------------------------------")

monthly_vol = avocado_df_no_west.copy()
monthly_vol['Month'] = monthly_vol['Month'] = monthly_vol['Date'].dt.month
monthly_vol['Year'] = monthly_vol['Year'] = monthly_vol['Date'].dt.year

#Group them thangs
timeline = monthly_vol.groupby(["Year","Month"]).agg({
    'Total Volume': 'sum'
}).reset_index()

#Plot that Timeline
plt.figure(figsize=(12, 6))
sns.lineplot(data=timeline, x=pd.to_datetime(timeline[['Year', 'Month']].assign(DAY=1)), y='Total Volume', marker='o', color='orange')
plt.title('Timeline of Total Volume of Avocados Sold Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Volume', fontsize=14)
plt.grid(alpha=0.75)
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.savefig('timeline_total_volume_over_time.png')
print("Timeline plot saved as 'timeline_total_volume_over_time.png'")
#plt.show()
plt.close()

#Highest average
month_avg_volume = monthly_vol.groupby('Month')['TotalVolume'].mean()
highest_volume_month = int(month_avg_volume.idxmax())
month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

print(f"\nObservation: The month with the highest average total volume of avocados sold is {month_names[highest_volume_month]} with an ")
print("average volume of {month_avg_volume.max():,.0f} units. This could be due to seasonal factors such as increased demand during ")
print("warmer months or holidays. It could have also just been a good month for guac")


#Thats a wrap babyy
print("\n")
print("END OF ASSIGNMENT 3.1")