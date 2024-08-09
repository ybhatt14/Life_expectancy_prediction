# -*- coding: utf-8 -*-
"""Group10_life_expectancy_prediction.ipynb


#Name
Arlene D’costa 
Yugmi Bhatt    
Sowmya Borra   

## Life expectancy data - Exploration & Modeling

### Table of contents

1. [Understanding data](#understanding)
2. [Preprocessing data](#preprocessing)
3. [Statistical summary](#statisticalSummary)
4. [Data visualization](#dataViz)
5. [Baseline model](#baseline)
6. [Linear Regression model](#mlModel)
7. [Support vector machine](#svr)
8. [K-Nearest Neighbor](#knn)
8. [Tree based models and ensembles](#treeModels)
10. [Feedforward neural network](#fnn)

# 1. Understanding Data <a name="understanding"></a>
"""

# !pip install rfpimp dtreeviz

# Commented out IPython magic to ensure Python compatibility.
# Data manipulation libraries
import pandas as pd
import numpy as np
# Data visuslization and specialty libraries
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from seaborn import heatmap, jointplot
import plotly.express as px
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
import rfpimp
# Machine learning libraries
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
import dtreeviz
import warnings
# Ignore all warnings
warnings.filterwarnings('ignore')



from google.colab import drive
drive.mount("/content/drive", force_remount=True)

path =  "/content/drive/MyDrive/DADT"

# Creating pandas DataFrame
life_expectancy_df = pd.read_csv(f"{path}/Life Expectancy Data.csv")

# Observing the first 5 rows of the DataFrame
life_expectancy_df.head()

"""**1.2. Dataset size**"""

# Printing number of rows, columns
life_expectancy_df.shape

"""It can be seen that the dataset has 2938 rows and 22 columns.

**1.3. Data types**

Checking the data types of the dataset columns,
"""

life_expectancy_df.info()

"""It can be seen that this dataset contains data columns of type float64, int64 and object.

**1.4. Basic summary statistics**

We can get the basic summary statistics for every numerical column using the pandas describe method. This provides the mean, standard deviation, quartiles, minimum and maximum values for the data columns.

This 'count' row gives us the counts of non-missing values in the columns.
"""

life_expectancy_df.describe()

"""When checking the above summary statistics, it can be seen that the values for the 'BMI' column are unrealistic.

Therefore this column must be dropped before continuing,
"""

# Make copy of the dataframe before making changes
life_expectancy_data = life_expectancy_df.copy()

# Drop column 'BMI'
life_expectancy_data = life_expectancy_data.drop(' BMI ', axis=1)

"""# 2. Preprocessing <a name="preprocessing"></a>

The data is in First Normal Form (1NF). The dataset has a flat structure, each row is uniquely identifiable, every column represents only one feature of the dataset and every column contains atomic values.
"""

# Printing the first 5 rows of the DataFrame
life_expectancy_data.head()

"""Getting rid of the white spaces in the column names,"""

# Make copy of the dataframe before making changes
life_expectancy = life_expectancy_data.copy()

# Dictionary mapping old column names to new column names
new_column_names = {
    'Country': 'country',
    'Year': 'year',
    'Status': 'status',
    'Life expectancy ': 'life_expectancy',
    'Adult Mortality': 'adult_mortality',
    'infant deaths': 'infant_deaths',
    'Alcohol': 'alcohol',
    'percentage expenditure': 'percentage_expenditure',
    'Hepatitis B': 'hepatitis_b',
    'Measles ': 'measles',
    'under-five deaths ': 'under_five_deaths',
    'Polio': 'polio',
    'Total expenditure': 'total_expenditure',
    'Diphtheria ': 'diphtheria',
    ' HIV/AIDS': 'HIV_AIDS',
    'GDP': 'GDP',
    'Population': 'population',
    ' thinness  1-19 years': 'thinness_1-19_years',
    ' thinness 5-9 years': 'thinness_5-9_years',
    'Income composition of resources': 'income_composition_of_resources',
    'Schooling': 'schooling',
}

# Rename columns
life_expectancy.rename(columns=new_column_names, inplace=True)

# Check column names
life_expectancy.columns

"""### 2.1. Missing data analysis <a name="missingData"></a>

Getting the counts of missing data in every column,
"""

life_expectancy.isnull().sum()

"""Now that the missing value counts have been obtained, they can be analysed further.

**2.1.1. missingno matrix visualization**

missingno is a library that allows users to create useful visualizations to study missing data values in a dataset.

Using the missingno library to create a visualization to look for patterns in the missing data,

The colored bars show non-missing values while the white bars show missing values. This can be useful to look for clusters of missing data in the dataset and spot any patterns.
"""

# Create the missingno matrix
msno.matrix(life_expectancy.sample(800))

# Show the plot
plt.show()

"""We has only used 800 samples (less than half) of the dataset to try and spot any clear pattern.

The columns containing 'country', 'year', 'status', 'infant_deaths', 'percentage_expenditure', 'measles', 'under_five_deaths' and 'HIV_AIDS' appear to be complete, while the other columns show varying levels of spottiness.

The similar missing data patterns in 'life_expectancy' and 'adult_mortality' columns may mean that there is some reliationship in the missing data between them. The 'polio' and 'diphtheria' columns show some similarity in missingness, and the 2 'thinness' columns also show a similar missingness pattern.

Further, the missingness pattern in the 'alcohol' and 'total_expenditure' column shows some similarity too.

**2.1.2. missingno bar visualization**

The missingno library can be used again to visualize the percentages of missing data.
Shorter bars denote a higher missing percentage.
"""

# Create the missingno barplot to get percents of nonmissing data
msno.bar(life_expectancy)

# Show the plot
plt.show()

"""It can be seen that the columns containing values for 'population', 'GDP' and 'hepatitis_b' have the largest percentages of missing data.

**2.1.3. missingno heatmap visualization**

Before attemptng to handle the missing values, we will create a missingno heat map to find if there are correlations among columns with missing data.

The numbers in the missingno heat map denote the correlation coefficients between the missing values of 2 variables. The correlation coefficient is a measure of how strongly the missingness of a variable is connected to the missingness of another variable.
"""

# Create heat map with missingno
msno.heatmap(life_expectancy)

# Show the plot
plt.show()

"""The above visualization shows that there is a perfect correlation between the missingness of 'adult_mortality' and 'life_expectancy' columns. It can also be observed that a strong correlation exists between the missingness of 'alcohol' and 'total_expenditure' columns.

There also appears to be a perfect correlation between the missingness of the 2 'thinness' columns.

### 2.2. Fixes to missing data <a name="missingDataFixes"></a>

The dataset contains data for countries for 16 years (2000 - 2015). Every country must have 16 data rows each.

The suthor will first remove any countries that have less than 16 data rows.
"""

# Get value counts for the 'country' column
rows_per_country = life_expectancy['country'].value_counts()

# Get the countries that have less than 16 data rows
countries_less_data = rows_per_country[rows_per_country < 16].index

# Remove countries that have less than 16 data rows and store result in a new DataFrame
life_exp_df = life_expectancy[~life_expectancy['country'].isin(countries_less_data)]

# Checking the null value counts after removing countries with less than 16 data rows.
life_exp_df.isnull().sum()

"""**2.2.1. Alcohol & total expenditure**

As observed in the missingno heatmap visualization, there is a strong positive correlation between the missingness of 'alcohol' and 'total_expenditure' columns.

Since the dataset has 16 data rows for each country, the NaN values in the 'alcohol' and 'total_expenditure' columns can be replaced with the mean values of column data belonging to the respective countries.
"""

# Calculate the mean 'Alcohol' value for each country
mean_alcohol_val = life_exp_df.groupby('country')['alcohol'].mean()
# Calculate the mean 'Total expenditure' value for each country
mean_tot_exp_val = life_exp_df.groupby('country')['total_expenditure'].mean()

"""We can now create a function that replaces the NaN values in a column with the mean attribute value for every country,"""

# Function to replace a column's NaN values with the mean attribute value for that country
def replace_nan_with_mean(mean, df, column_name):
    """
    Replace NaN data with mean
    Ex: if for a year for UK, GDP data is missing, replace that data with mean GDP of UK
    :param mean: mean for all countries
    :param df: data frame to mutate
    :param column_name: name of the column to check
    """
    # Replacing the NaN values in the given column by iterating over every row
    for index, row in df.iterrows():
        # Check if column_name value is a NaN
        if pd.isnull(row[column_name]):
            # Get the correct country name
            country = row['country']
            # Replace the NaN value with the mean '?' value of that country
            df.loc[index, column_name] = mean[country]

"""Using the new function to replace NaN values in 'alcohol' and 'total_expenditure' columns with the mean attribute value of that country,"""

# "Alcohol" column
replace_nan_with_mean(mean_alcohol_val, life_exp_df, 'alcohol')

# "Total expenditure" column
replace_nan_with_mean(mean_tot_exp_val, life_exp_df, 'total_expenditure')

# Remove any rows that still have NaN values in alcohol and total_expenditure columns
life_exp_df3 = life_exp_df.dropna(subset=['alcohol', 'total_expenditure'])

"""**2.2.2. Hepatitis B**

As noted in a previous section (Missing data analysis), the 'Hepatitis B' column hsa a considerable percentage of missing data (roughly 18.8 percent). The missingno heatmap showed no interesting correlation between the missingness of the 'Hepatitis B' column and other columns. The missingno matrix visualization also showed no clear pattern of missingness for the 'Hepatitis B' column.

We can try to replace the NaN values in this column with the mean 'Hepatitis B' value of its respective country,
"""

# Calculate the mean 'Hepatitis B' value for each country
mean_hepB_val = life_exp_df3.groupby('country')['hepatitis_b'].mean()

# Replacing the NaN values in 'Hepatitis B' column with the mean by iterating over every row
replace_nan_with_mean(mean_hepB_val, life_exp_df3, 'hepatitis_b')

# Calculating the remaining NaN value counts for 'Hepatitis B' column
life_exp_df3['hepatitis_b'].isnull().sum()

"""It can be seen that there are 144 rows that still have NaN values in the 'hepatitis_b' column. We could drop these rows, but first we will find out if the 'hepatitis_b' and 'life_expectancy' columns share a notable correlation,"""

# Computing the correlation between 'hepatitis b' and 'life expectancy' columns
corr_val = life_exp_df3[['life_expectancy', 'hepatitis_b']].corr()
corr_val

"""It can be seen that the correlation between the 2 columns is insignificant, and the 'hepatitis_b' column can be dropped.

Dropping the 'hepatitis_b' column,
"""

# Drop column 'Hepatitis B'
life_exp_df4 = life_exp_df3.drop('hepatitis_b', axis=1)

# Checking the new percentages of missing column values,
life_exp_df4.isnull().mean() * 100

"""**2.2.3. Polio**

Handling missing 'polio' data,
"""

# Calculate the mean 'polio' value for each country
mean_polio_val = life_exp_df4.groupby('country')['polio'].mean()

# Use replace_nan_with_mean function to replace NaN 'Polio' values with the mean 'Polio' of its respective country
replace_nan_with_mean(mean_polio_val, life_exp_df4, 'polio')

# Calculating the remaining NaN value count percentages
life_exp_df4.isnull().mean() * 100

"""It can be seen that there are no more missing 'Polio' values.

**2.2.4. GDP**

14% of values in the 'GDP' column are NaN values. We can attempt to replace some of these values with the mean 'GDP' values of their respective countries,
"""

# Calculate the mean 'GDP' value for each country
mean_gdp_val = life_exp_df4.groupby('country')['GDP'].mean()

# Use replace_nan_with_mean function to replace NaN 'GDP' values with the mean 'GDP' of its respective country
replace_nan_with_mean(mean_gdp_val, life_exp_df4, 'GDP')

# Dropping rows that still have missing values for 'GDP'.
life_exp_df5 = life_exp_df4.dropna(subset=['GDP'])

# Getting the remaining NaN value counts
life_exp_df5.isnull().sum()

"""**2.2.5. Population**

There are 228 NaN values in the 'Population' column. We can attempt to replace some of the NaN values with the mean 'Population' values of their respective countries,
"""

# Calculate the mean 'Population' value for each country
mean_pop_val = life_exp_df5.groupby('country')['population'].mean()

# Use replace_nan_with_mean function to replace NaN 'Population' values with the mean 'Population' of its respective country
replace_nan_with_mean(mean_pop_val, life_exp_df5, 'population')

# Calculating the remaining NaN value count
life_exp_df5['population'].isnull().sum()

"""There are still 224 NaN values in the 'Population' column. Before removing any data, we can check the correlation value between 'Life expectancy' and 'Population',"""

# Computing the correlation between ''population' and 'life_expectancy' columns
corr_val = life_exp_df5[['life_expectancy', 'population']].corr()
corr_val

"""The correlation value is roughly -0.02, which means that the correlation between these 2 columns is not significant, and so the Population column can be dropped."""

# Dropping 'Population' column
life_exp_df6 = life_exp_df5.drop('population', axis=1)

life_exp_df6.head()

"""**2.2.6. Diphtheria**

Checking which rows have missing values for the 'diphtheria' column,
"""

life_exp_df6[life_exp_df6['diphtheria'].isnull()].head()

"""We can replace these NaN values with the mean 'diphtheria' values of their respective countries,"""

# Calculate the mean 'Diphtheria' value for each country
mean_dip_val = life_exp_df6.groupby('country')['diphtheria'].mean()

# Use replace_nan_with_mean function to replace NaN 'Diphtheria ' values with the mean 'Diphtheria ' of its respective country
replace_nan_with_mean(mean_dip_val, life_exp_df6, 'diphtheria')

# Calculating the remaining NaN value count
life_exp_df6['diphtheria'].isnull().sum()

"""Now there are no more NaN values in the 'diphtheria ' column.

**2.2.7. Thinness**

Checking which rows have missing values for the 'thinness  1-19 years' column,
"""

life_exp_df6[life_exp_df6['thinness_1-19_years'].isnull()]

"""All 'thinness' values for Sudan are missing, so we are unable to calculate a mean value to replace this data. The Sudan rows will need to be removed."""

# Store result in new DataFrame
life_exp_df7 = life_exp_df6[life_exp_df6['country'] != 'Sudan']

# Check if all missing values have been dealt with,
life_exp_df7.isnull().mean() * 100

"""Sanity check"""

#  isna() returns a dataframe with boolean values for all cells
#  now call any() to check if any column has missing values
#  final call to any() determines if any cell in the dataframe has missing values

life_exp_df7.isna().any().any() # Output 'False' means that there are no missing values

"""### 2.3. Remove duplicate data <a name="duplicateData"></a>

Check for any duplicate data records,
"""

# Finding duplicate rows
life_expectancy[life_expectancy.duplicated()]

"""### 2.4. Handle outlier data <a name="outlierData"></a>

We can print out the summary statistics to check for extreme or unrealistic values,
"""

# Summary statistics
life_exp_df7.describe()

"""Creating a function to get outlier data of a column (feature),"""

### Adapted from https://www.kaggle.com/discussions/general/218717#1200703

def get_outlier_data(df, feature):
    """
    Get outliers for a given feature (column name)
    :param df: Dataframe to check
    :param feature: column name of the feature in df
    :return: lower_quartile, upper_quartile, outlier_rows as a tuple
    """
    # Calculate the interquartile range
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    # Set lower and upper limits for outliers
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Get outliers
    outliers = df[(df[feature] < lower) | (df[feature] > upper)]
    return lower, upper, outliers

"""We can now use the above function to look for outliers in the data.

Getting the outlier values in the Life expectancy column,
"""

# Get outlier data points for life expectancy column using the newly created function
lower_life_exp, upper_life_exp, outliers_life_exp = get_outlier_data(life_exp_df7, 'life_expectancy')

print(lower_life_exp, upper_life_exp)
outliers_life_exp

"""The custom function can be modified so that extreme values will be replaced by the upper or lower bound value."""

# Function to replace outliers
def replace_outliers(df, feature):
    """
    Replace outliers of df with lower and upper bound of the data
    :param df: data frame to mutate
    :param feature: feature's column name
    """
    # Calculate the interquartile range
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    # Set lower and upper limits for outliers
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Replace small outlier values with lower bound
    df.loc[df[feature] < lower, feature] = lower
    # Replace the large outlier values with the upper_bound
    df.loc[df[feature] > upper, feature] = upper

"""We will create a new DataFrame before replacing the outliers in the data,"""

# Create copy of DataFrame
life_exp_df8 = life_exp_df7.copy()

# Replace outliers in the 'life_expectancy' column
replace_outliers(life_exp_df8, 'life_expectancy')

"""We can check for outliers in the new DataFrame's 'Life expectancy' column by using the first function,"""

# Get outlier data points for life expectancy column using the newly created function
lower_life_exp, upper_life_exp, outliers_life_exp = get_outlier_data(life_exp_df8, 'life_expectancy')

print(lower_life_exp, upper_life_exp)
outliers_life_exp

"""It can be seen that there are no more outlier data points in the 'life_expectancy' column.

We can now use the modified function to find and replace outlier data points in the other numeric columns as well.
"""

# List of column names we can apply the outlier replacing function on
columns_to_iterate = ['adult_mortality', 'infant_deaths', 'alcohol', 'percentage_expenditure',
                      'measles', 'under_five_deaths', 'polio', 'total_expenditure', 'diphtheria',
                      'HIV_AIDS', 'GDP', 'thinness_1-19_years', 'thinness_5-9_years',
                      'income_composition_of_resources', 'schooling']

# Iterating over the remaining numeric columns
for col in columns_to_iterate:
    # Apply function to replace outliers
    replace_outliers(life_exp_df8, col)

"""Now there are no more outliers in the data.

### 2.5. Normalize data <a name="normalizeData"></a>

We will normalize the numeric columns which will rescale the data to values between 0 and 1. Normalizing the data will preserve the shape of the original distribution while having all the features on a similar scale.
"""

# Create a copy of the DataFrame before normalizing
life_exp_df9 = life_exp_df8.copy()

# Select numeric columns that need to be normalized
features = ['life_expectancy', 'adult_mortality', 'infant_deaths', 'alcohol', 'percentage_expenditure',
            'measles', 'under_five_deaths', 'polio', 'total_expenditure', 'diphtheria',
            'HIV_AIDS', 'GDP', 'thinness_1-19_years', 'thinness_5-9_years',
            'income_composition_of_resources', 'schooling']

# Using MinMaxScaler from sklearn library to normalize data
# Create a MinMaxScaler object
scaler = MinMaxScaler()
# Normalize the features
life_exp_df9[features] = scaler.fit_transform(life_exp_df9[features])

# Print the normalized dataset
life_exp_df9.head()

"""### 2.6. Key series identification <a name="keySeriesId"></a>

The key series of this dataset are the columns that contain the most useful data for predicting life expectancy. We has used a seaborn heatmap to find out which columns show a notable correlation with the 'Life expectancy' column.
"""

# Create seaborn heat map
fig, ax = plt.subplots(figsize=(8, 8))
ax = heatmap(life_exp_df9.corr(numeric_only=True), fmt=".1f", annot=True, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1)
plt.show()

"""We can drop the columns that share a collinearity value between 0.6 and -0.4 with the 'Life expectancy' column and also categorical columns such as Country and status which do not provide much meaning for life expectancy"""

# Dropping columns that have less than 0.5 collinearity with Life expectancy' and storing result in new DataFrame
# Also dropping categorical features for the present
life_exp_df10 = life_exp_df9.drop(
    ['country', 'year', 'status', 'alcohol', 'measles', 'total_expenditure'], axis=1)

"""### 2.7. Handle multicollinearity <a name="multicollinearity"></a>

It can be seen that some of the columns that are collinear with 'Life expectancy' are also highly collinear with each other. Such highly correlated columns can make a model difficult to interpret.

To visualize multicollinearity and decide what features we really need, we can create a dependence heat map by using the rfpimp package,
"""

# FROM: O'Reilly Machine Learning Pocket Reference by Matt Harrison

# (Passing in a copy of the DataFrame, because the rfpimp library sometimes makes changes to the data)
rfpimp.plot_dependence_heatmap(rfpimp.feature_dependence_matrix(life_exp_df10.copy()),
                               value_fontsize=8,
                               label_fontsize=8,
                               figsize=(8,8)
                              )

fig = plt.gcf()

"""From studying the above dependencies we can deduce that a few columns need to be removed.

We can remove either of the 'infant_deaths' or 'under_five_deaths' columns. We can also remove either of the 'GDP' or 'percentage_expenditure' columns. One of the 'thinness' columns can be removed. One of either 'schooling' or 'income composition of resources' columns can also be removed. We can remove one of 'diptheria' or 'polio' as well.

Removing columns that can disrupt feature importance,
"""

# Remove columns and store result in new DataFrame
life_exp_df11 = life_exp_df10.drop(['infant_deaths', 'percentage_expenditure', 'thinness_5-9_years',
                                    'diphtheria', 'income_composition_of_resources'], axis=1)

# Show first 5 rows
life_exp_df11.head()

"""# 3. Statistical summary <a name="statisticalSummary"></a>

Now that the key series of the dataset have been identified, a statistical analysis can be performed.

### 3.1. Measures of central tendency <a name="centralTendency"></a>

**A. Mode** <a name="mode"></a>

Getting most frequent scores in every column,
"""

life_exp_df11.mode()

"""**B. Mean** <a name="mean"></a>

Getting the mean values for each column,
"""

# Mean values of normalized key series
life_exp_df11.mean()

"""Visualizing the mean values for each column,"""

mean_vals = life_exp_df11.mean()

# Create a bar plot
mean_vals.plot(kind='bar', figsize=(5, 5))

# Set plot title and labels
plt.title('Mean Values of key series data')
plt.xlabel('Columns')
plt.ylabel('Mean')

# Show the plot
plt.show()

"""It can be seen that 'Polio' appears to have the highest mean value of the columns, and 'HIV_AIDS' has the lowest.

**C. Median (Second quartile)** <a name="median"></a>

Calculating the median value for every column,
"""

# Median values of normalized key series
life_exp_df11.median()

"""Visualizing the median values for each column,"""

median_vals = life_exp_df11.median()

# Create a bar plot
median_vals.plot(kind='bar', figsize=(5, 5))

# Set plot title and labels
plt.title('Median Values of key series data')
plt.xlabel('Columns')
plt.ylabel('Median')

# Show the plot
plt.show()

"""It can be seen that 'Polio' also has the highest median value among the columns.

As 'Polio' has the highest median and mean, this means that this column has higher values than the other columns in the normalized dataframe.

### 3.2. Measures of spread <a name="spread"></a>

**A. Range** <a name="range"></a>

Calculating the range value for every column,
"""

range_vals = life_exp_df11.max(axis=0) - life_exp_df11.min(axis=0)

range_vals

"""**B. Quartiles** <a name="quartiles"></a>

Getting lower quartile values for every column,
"""

life_exp_df11.quantile(0.25)

"""Getting upper quartile values for every column,"""

life_exp_df11.quantile(0.75)

"""The overall frequency distribution of the variables, including quartile information can be visualized using a box plot,"""

# Create box plot to show frequency distribution summary
sns.boxplot(data=life_exp_df11, orient="h")
# Set title
plt.title('Frequency distribution of columns')
plt.xlabel('Values')
plt.show()

"""The colored boxes contain the central 50% of data. The vertical line in every box is the median value for that column.
The tails show the values in the lower and upper quartiles.

This box chart confirms the observations made previously, that 'Polio' column has the highest values.

**C. Variance** <a name="variance"></a>

Calculating the variance value for every column,
"""

# Variance values
life_exp_df11.var()

"""**D. Standard deviation** <a name="stdDev"></a>

Calculating the standard deviation value for every column,
"""

# Standard deviation values of normalized columns
life_exp_df11.std()

"""**E. Skewness** <a name="skewness"></a>

Creating a histogram to visuslize distribution of 'Life expectancy' data.
"""

# Create seaborn histogram
sns.histplot(data=life_exp_df11, x="life_expectancy", kde=True, bins=31)

# Set plot title
plt.title('Normalized Life expectancy distribution')
plt.show()

"""This is an assymetric distibution - a majority of the observations appear to the right of the mean. The left tail is longer than the right. Therefore this is a distribution of a left skew (negative skewness).

The calculation of the skewness value should return a negative value,
"""

# Using pandas skew to calculate skew value
life_exp_df11['life_expectancy'].skew()

"""Similarly, we can calculate skewness values for all the key series in the data,"""

life_exp_df11.skew()

"""It can be seen that the Life expectancy, Polio and Schooling columns have negative skew values. Most of the data in these columns appears to the left of the mean. The rest of the columns have positive skewness.

**F. Kurtosis** <a name="kurtosis"></a>

The kurtosis value informs us of the shape, the peak nature and the tails of the distribution. This value can be used to quantify how much the distribution has deviated from a normal distribution.

Calculating kurtosis values for the columns,
"""

life_exp_df11.kurtosis()

"""It can be seen that most of the columns show negative kurtosis. This means that these distributions are platykurtic - they contain fewer extreme values and the data is more spread out.

The 'adult_mortality', 'polio' and 'thinness_1-19_years' columns show positive kurtosis. This means that they are leptokurtic distributions. They have a more peaked shape than a normal distribution and have heavier tails.

# 4. Data visualization <a name="dataViz"></a>

Visualizing the key data series using a seaborn kde plot,
"""

# Create seaborn kde plot
sns.kdeplot(data=life_exp_df11, bw_adjust=2, cut=0)
# Show plot
plt.xlabel('Values')
plt.show()

"""The kde plot shows the distribution of observations for the key series. It can be seen that 'Adult Mortality', 'under-five deaths', 'HIV/AIDS', 'GDP' and 'thinness  1-19 years' columns are positively skewed while the other variables are negatively skewed.

We can also use a seaborn pair plot to visualize relationships between the key series,
"""

# Create seaborn pair plot
sns.pairplot(life_exp_df11)
plt.show()

"""The pairplot allows us to spot interesting relationships between variables. For instance, here it can be seen that countries with higher life expectancy also have a higher mean number of years of schooling. And when the HIV/AIDS deaths in children is higher, life expectancy generally becomes lower. Life expectancy also shows some positive collinearity with 'GDP'. Furthermore 'Adult mortality' appears to be negatively correlated with 'Life expectancy.'

#### A. Adult mortality

We can create a density plot to check the distribution of the 'Adult Mortality' data. We will use the data from before normalization to look for interesting observations.
"""

# Density plot to visualize distribution of 'Adult Mortality'
sns.kdeplot(data=life_exp_df8, x='adult_mortality')

# Set plot title and labels
plt.title('Adult Mortality density plot')
plt.xlabel('Adult Mortality')
plt.ylabel('Density')

# Show plot
plt.show()

"""It can be seen that most of the values appear around 100, and the mean value is roughly 137 (as seen in the summary statistics below)."""

# Summary statistics from before data was normalized
life_exp_df8.describe()

"""A mean value of 137 means that out of every 1000 people aged 15 years old, 137 of them can be expected to die before reaching the age of 60.

We can create a seaborn jointplot to visualize the relationship between 'Life expectancy' and 'Adult Mortality' by 'Status',It can be seen that 'Adult Mortality' has a strong negative correlation with 'Life expectancy'.
"""

# Create a jointplot
sns.jointplot(data=life_exp_df8, x='adult_mortality', y='life_expectancy', hue='status', palette='dark')

# Display plot
plt.show()

"""It can easily be observed that developed countries have higher life expectancy values and lower adult mortality values. We can examine this plot again, this time adding a linear regression fit and univariate KDE curves,"""

# Create a jointplot with linear regression
sns.jointplot(data=life_exp_df8, x='adult_mortality', y='life_expectancy', kind='reg', scatter_kws={'s': 5})

# Display plot
plt.show()

"""It can be seen that for the most part,the general pattern here is that life expectancy increases when adult mortality is low. As adult mortality increases, life expectancy decreases.

It can be seen that there is also a subset of data where life expectancy is low even when adult mortality is low. We can check the data from these countries to attempt to find the reason,
"""

# Using 25% quantile values from summary stats table to get rows where both life expectancy and adult mortality are low
life_exp_df8[(life_exp_df8['adult_mortality'] < 71) & (life_exp_df8['life_expectancy'] < 63)].head()

"""It can be seen that the countries that have both low adult mortality and low life expectancy have a comparatively high number of infant deaths and under-five deaths (see summary stats printed previously). The high numbers of infant deaths and under-five deaths could lower the calculated average life expectancy of the country.

This would explain how these countries have both low life expectancy and low adult mortality values.

It can also be seen that these countries have notably higher counts of measles cases, higher HIV/AIDS deaths in children, and a lower number of mean years of schooling.

#### B. Schooling

'Schooling' is another key series in our data. We can use a violin plot to visualize the distribution of this data,
"""

# Violin plot to visualize distribution of 'Schooling' data by the 'Status'
sns.violinplot(data=life_exp_df8, x='status', y='schooling', palette='pastel')

# Set plot title and labels
plt.title('Distribution of Schooling data by Status')
plt.xlabel('Status')
plt.ylabel('Schooling')

# Display the plot
plt.show()

"""It can be seen that for developed countries, the mean number of years of schooling is higher than for developing countries. The width of the violins suggest that developed countries have a much higher density of persons that complete over 15 years of schooling than developing countries.

The symmetrical shapes of the violins also suggest that the data is distributed around the median in an approximately symmetrical manner.

We can use another seaborn plot to visualize the reliationship between Schooling and life expectancy,
"""

# Create seaborn lineplot
sns.lineplot(data=life_exp_df8, x="schooling", y="life_expectancy", hue="status", palette='pastel')

# Display the plot
plt.title('Life expectancy vs mean schooling years')
plt.show()

"""As noted in the pair plot observations, this plot confirms that a higher number of mean years of schooling is generally related to higher life expectancy values.

#### C. HIV/AIDS

The seaborn pair plot visualized previously suggested that HIV/AIDS deaths in children is correlated negatively to life expectancy. We can use a strip plot visualization to understand the distribution of this key series data,
"""

# Create a stripplot to visualize the distribution of HIV/AIDS by country Status
sns.stripplot(data=life_exp_df8, x='status', y='HIV_AIDS', hue='status', palette='Set3')

plt.title('HIV/AIDS deaths in children by Status')
# Show the plot
plt.show()

"""It can be seen that seen that the vast majority of HIV/AIDS death in children occur in countries with 'Developing' status. We can create another visualization to study the relationship between Life expectancy and HIV/AIDS deaths in children,"""

# Create seaborn lmplot
sns.lmplot(data=life_exp_df8, x="HIV_AIDS", y="life_expectancy", hue="status", palette='Set3')
# Create seaborn rug plot
sns.rugplot(data=life_exp_df8, x="HIV_AIDS", y="life_expectancy", hue="status", palette='Set3')

plt.title('Life expectancy vs HIV/AIDS deaths in children')
# Show the plot
plt.show()

"""It can be seen that the average life expectancy value shows a generally decreasing pattern as the number of HIV/AIDS deaths in children increase.

#### D. Under-five deaths

Next we can visualize the 'under-five deaths' column to understand its data distribution better,
"""

# Displot for visualizing the distribution of under-five deaths by developed/developing status
sns.displot(data=life_exp_df8, x='under_five_deaths', hue='status', kde=True, palette='Set1')

# Plot title and labels
plt.title('Under-five distribution by Status')
plt.xlabel('Under-five deaths')
plt.ylabel('Density')
plt.show()

"""It can be observed that the number of deaths in children under age 5 is very low in countries with Developed status in comparison with countries having Developing status. Also, the density of under-five deaths becomes notably high as the value increases past 50. This has happened because we previously capped higher values in this column at 57.5

Printing out the data rows makes this more clear.
"""

# Getting the first rows where under-five death count is greater than 50
life_exp_df8[life_exp_df8['under_five_deaths'] > 50].head()

"""We can create a visualization to see the impact of under-five deaths on the average life expectancy value of a country"""

# Create line plot
sns.lineplot(data=life_exp_df8, x='under_five_deaths', y="life_expectancy", hue="status", palette='Set1')

# Plot title and labels
plt.title('Life expectancy vs under-five deaths')
plt.xlabel('Under-five deaths')
plt.ylabel('Life expectancy')
plt.show()

"""It can be observed that the average life expectancy value generally decreases as the under-five death count increase.

#### E. Polio

The next key series data to visualize is Polio.
"""

# Violin plot to visualize distribution of 'Polio' data by the 'Status'
sns.violinplot(data=life_exp_df8, x='polio', y='status', palette='Set2')

# Set plot title and labels
plt.title('Distribution of Polio immunization percentage data by Status')
plt.ylabel('Status')
plt.xlabel('Polio immunization %')

# Display the plot
plt.show()

"""It can be seen by the width of the violins that the density of Polio immunization for children over 1 year old is notably higher for developed countries over developing countries.

Visualizing Polio immunization percentage against life expectancy,
"""

# Create seaborn lmplot
sns.lmplot(data=life_exp_df8, x="polio", y="life_expectancy", hue="status", scatter_kws={'s': 10}, palette='Set2')
# Create seaborn rug plot
sns.rugplot(data=life_exp_df8, x="polio", y="life_expectancy", hue="status")

plt.title('Life expectancy vs Polio immunization percentage')
# Show the plot
plt.show()

"""This visualization supports the correlation heatmap seen in a previous section, which showed these 2 variables to have a moderately strong correlation value of 0.6.

The percentage of Polio immunization does not have a notable effect on life expectancy for developed countries. However for developing countries, the average life expectancy value appears to generally increase as the Polio immunization percentage increases.

#### F. GDP

The next key series to visualize is GDP,
"""

# Displot for visualizing GDP by developed/developing status
sns.displot(data=life_exp_df8, x='GDP', hue='status', kde=True, palette='dark')

# Plot title and labels
plt.title('GDP distribution by Status')
plt.xlabel('GDP')
plt.ylabel('Density')
plt.show()

"""It can be seen that the density of countries with very low GDP values is much higher the density of countries having a high GDP value.

Visualizing the relationship between GDP and life expectancy,
"""

# Create seaborn lmplot
sns.lmplot(data=life_exp_df8, x="GDP", y="life_expectancy", hue="status", scatter_kws={'s': 5}, palette='dark')
# Create seaborn rug plot
sns.rugplot(data=life_exp_df8, x="GDP", y="life_expectancy", hue="status")

plt.title('Life expectancy vs GDP')
# Show the plot
plt.show()

"""It can be observed that life expectancy appears to increase as a country's GDP increases.

#### G. thinness 1-19 years

This column refers to the percentage thinness of children aged between 1 and 19 years.
"""

# Create a stripplot to visualize the distribution of HIV/AIDS by country Status
sns.stripplot(data=life_exp_df8, x='status', y= 'thinness_1-19_years', hue='status', palette='Set2')

plt.title('thinness in children by country status')
# Show the plot
plt.show()

"""It can be seen that Developing countries have a higher percentage thinness in children than Developed countries.

Visualizing the relationship between child thinness percentage and life expectancy.
"""

# Create seaborn lineplot
sns.lineplot(data=life_exp_df7, x='thinness_1-19_years', y="life_expectancy", hue="status", palette='Set2')

# Display the plot
plt.title('Life expectancy vs child thinness percentage')
plt.show()

"""Life expectancy appears to decrease with child thinness percentage, but only up to a point. When the child thinness percentage value approaches 15, the average life expectancy increases. Past this point, the life expectancy appears to settle in a value range from about 60 to 75.

**H. An overall look**

Now that the key data series have been visualized, we can look at how life expectancy has increased over the years.

Creating a bar plot visualization to show how the average life expectancy has increased over the years 2000 to 2015,
"""

# Calculate the mean life expectancy value by Year
mean_life_exp = life_exp_df8.groupby('year')['life_expectancy'].mean().reset_index()

# Bar plot
sns.barplot(data=mean_life_exp, x='year', y='life_expectancy', palette='flare')
plt.title('Mean Life expectancy value by Year')
plt.xticks(rotation=70)
plt.tight_layout()

plt.show()

"""Below is an interactive visualization that shows the distribution of life expectancy data,"""

# plotly interactive histogram
fig = px.histogram(life_exp_df8,
                   x='life_expectancy',
                   color='status',
                   template='plotly_dark',
                   title='<b>Distribution of life expectancy data')
fig.show()

"""It can be seen that the most common life span count is around the mid 70s, and that the average life expectancy of Developed countries appears to be higher than that of Developing countries.

We can also created an animated visualization of the life expectancy of countries over the years 2000-2015,
"""

# plotly line plot
fig = px.line(life_exp_df8.sort_values(by='year'),
              x='year',
              y='life_expectancy',
              range_y=[35, 100],
              range_x=[2000, 2015],
              animation_frame='country',
              animation_group='year',
              color='country',
              markers=True,
              template='seaborn',
              title='<b>Life expectancy of countries from 2000-2015')
fig.show()

"""# 5. Baseline model <a name="baseline"></a>

A baseline regression model serves as a comparison for other models. We can use sklearn's DummyRegressor which will predict the mean value of the training set. The result score will be the r2 score (co-efficent of determination).
"""

# Initialize dummy regressor
dummy_reg_model = DummyRegressor()
dummy_reg_model

"""Arrange data into features matrix and target vector"""

# Dropping life expectancy column as it is the target variable
X_life_exp = life_exp_df11.drop('life_expectancy', axis=1)
print(X_life_exp.shape) # checking shape

# Creating target vector
y_life_exp = life_exp_df11['life_expectancy']
print(y_life_exp.shape) # checking shape

"""Setting up the training and test data,"""

# Leaving a third of the data out to test the model
X_train, X_test, y_train, y_test = train_test_split(X_life_exp, y_life_exp, test_size=1/3, random_state=42)

"""Fitting the baseline model to the data,"""

# Use the training datasets to fit the model
dummy_reg_model.fit(X_train, y_train)

# Print r2 score
dummy_reg_score = dummy_reg_model.score(X_test, y_test)
print(f"Baseline model score (r2): {dummy_reg_score:.5f}")

"""It can be seen that this model's performance is very poor.

# 6. Linear regression model<a name="mlModel"></a>

### 6.1. Building the model <a name="modelBuilding"></a>

**Feature selection**

Linear regression models assume that the prediction is a linear combination of the inputs.

The model works best when using features that are not strongly collinear with each other, but show correlation with the target variable. So we can use the dataframe that uses the key series and contains no collinear features

Features: 'Adult Mortality', 'under-five deaths', 'Polio', 'HIV/AIDS', 'GDP', 'thinness 1-19 years' and 'Schooling'
"""

# Printing first 5 rows of the processed data to see the features and label
life_exp_df11.head()

"""**Building linear regression model**"""

linear_reg_model = LinearRegression()

# Fitting the model to the data
# Use the training datasets to fit the model
linear_reg_model.fit(X_train, y_train)

"""**Apply model to unseen data**

Calculating r-squared (the coefficient of determination) to determine how well the regression performs on the unseen test data.
"""

# Calculating r-squared
lin_reg_r2 = linear_reg_model.score(X_test, y_test)

print(f"Linear reg score (r2): {lin_reg_r2:.5f}")

"""The result is roughly about 84%, which tells us that this model performs reasonably well on data it has not seen before. The result means that roughly 84% of the variability in 'Life expectancy' can be connected with a linear relationship with the model features.

It can be seen that the model fits the data and further preprocessing is not required.

### 6.2. Model validation <a name="validation"></a>

#### 6.2.1. Metrics assessment <a name="metricsAssessment"></a>

**A. Mean squared error (MSE)**

The first metric we can check is the mean squared error,
"""

# Predicting y values for unseen X data
y_pred = linear_reg_model.predict(X_test)

# Calculate mean squared error value
lin_reg_mse = mean_squared_error(y_test, y_pred)
print(f"Linear reg MSE: {lin_reg_mse:.5f}")

"""The model has a low mean squared error value, which validates the linear regression model. This low error value indicates that the predictions made by the model appear to have good precision and that the model appears to capture the relationships in the data satisfactorily.

**B. Mean absolute error (MAE)**

Another metric to check is the mean absolute error,
"""

# Calculate mean absolute error value
lin_reg_mae = mean_absolute_error(y_test, y_pred)
print(f"Linear reg MAE: {lin_reg_mae:.5f}")

"""It can be seen that the mean absolute error is low, which means that values predicted by the model are close to the true values in the dataset. Therefore, this metric has also validated the model.

#### 6.2.2. Cross validation <a name="crossValidation"></a>

A cross validation can be used to validate the results of the model. We will use random-fold validation. This will shuffle the data repeatedly and do random train/splits on the data.
"""

# From - O'Reilly Essential Math for Data Science by Thomas Nield pg. 190

# Using 10 splits will give us 10 scores
kfold = ShuffleSplit(n_splits=10, test_size=.33, random_state=7)
result_cross_val = cross_val_score(linear_reg_model, X_life_exp, y_life_exp, cv=kfold)

print(result_cross_val)

"""The result array shows the performance of our linear regression model on every randomized train-test split.

Looking at this array it can be seen that there is less variance among the values. The model has achieved almost the same performance across different splits of the data.

This means that the model is able to generalize well to different subsets of the data.

We can calculate the mean result to check it against our original model score.
"""

# Mean performance
result_cross_val.mean()

"""It can be seen that this result is extremely close to the model score (0.83730), which validates the performance of our linear regression model.

#### 6.2.3. Residual analysis <a name="residualAnalysis"></a>

A residual analysis can be performed to study the difference between the predicted and true values for the linear regression model.
"""

# Calculate the residuals
residuals = y_test - y_pred

# Plot the residuals against the predicted values
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Analysis")
plt.show()

"""We can also use Yellowbrick to create another residuals plot to visualize homoscedasticity and outliers."""

from yellowbrick.regressor import ResidualsPlot

fig, ax = plt.subplots(figsize=(6, 4))
res_plot_viz = ResidualsPlot(linear_reg_model)
res_plot_viz.fit(X_train, y_train)
res_plot_viz.score(X_test, y_test)
res_plot_viz.poof()

"""We can test for heteroscedasticity using the Breusch-Pagan test, to find out if the predictions are biased."""

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# Add a constant column for intercept
X = sm.add_constant(X_life_exp)
y = y_life_exp

model = sm.OLS(y, X)
results = model.fit()

# Obtain the residuals
residuals = results.resid

# Perform the Breusch-Pagan test
lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(residuals, X)

# Print the results
print("Breusch-Pagan test results:")
print(f"LM statistic: {lm:.5f}")
print(f"LM test p-value: {lm_p_value:.5f}")
print(f"F-statistic: {fvalue:.5f}")
print(f"F-test p-value: {f_p_value:.5f}")

"""Since both the LM test p-value and the F-test p-value are very small (close to 0), we reject the null hypothesis of homoscedasticity. The results suggest that the variance of the errors in our regression model is not constant across different levels of the independent variables, and so we can conclude that the predctions are biased. We can use alternative models to combat the problem of heteroscedasticity.

We can look at the slope and the intercept of the linear fit to the data,
"""

print(linear_reg_model.coef_) # slope
print(linear_reg_model.intercept_) # intercept

"""It can be seen that some of the coefficients are positive and some are negative.

This means that the model has considered the overall relationship between the features and the label. It has taken into account that some variables such as 'Schooling' and 'GDP' are positively correlated to 'Life expectancy' while other variables such as 'Adult mortality' and 'HIV/AIDS' are negatively correlated.

The larger the absolute value of the coefficient, the more impact it has on the model. We can visualize the coefficients using the YellowBrick library to understand which features contribute most to the linear regression.
"""

from yellowbrick.features import FeatureImportances

fig, ax = plt.subplots(figsize=(8, 4))
feature_importance_vis = FeatureImportances(linear_reg_model)
feature_importance_vis.fit(X_life_exp, y_life_exp)
feature_importance_vis.poof()

"""The visualization indicates that life expectancy of a country increases as the average number of schooling years increases, and that increasing HIV/AIDS rates will negatively impact the life expectancy.

### 6.3. Feature engineering <a name="featureEngineering"></a>

#### 6.3.1. Polynomial features <a name="polynomialFeatures"></a>

We can add more flexibility to the model by using polynomial features, which may result in a better fit.

We will use the grid search module from Scikit-Learn to attempt to find the best polynomial model.
"""

# Polynomial Regression code from O'REILLY Python Data Science Handbook by Jake VanderPlas

def PolynomialRegression(degree=2, **kwargs):
    """
    Create a pipeline for polynomial regression
    This is a class like function that constructs an object (This is why UpperCamelCase is used)
    :param degree: degree of polynomial features
    :param kwargs: options for linear regression
    :return: pipeline object
    """
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

param_grid = {'polynomialfeatures__degree': np.arange(1, 8),
              'linearregression__fit_intercept': [True, False]}

grid_search = GridSearchCV(PolynomialRegression(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_
print("Best Parameters: ", best_params)

"""These parameters will be used to fit the data to the new polynomial model."""

# Using the best model
polynomial_model = grid_search.best_estimator_
polynomial_model.fit(X_train, y_train)

"""Test the model using unseen data,"""

# Calculating r-squared
poly_score_r2 = polynomial_model.score(X_test, y_test)
print(f"Polynomial model score (r2): {poly_score_r2:.5f}")

"""**Evaluation**

We can see that the new model score is about 91%, which is a notable improvement from the previous model score of 84%.

Therefore this polynomial model is an improvement over the initial linear regression model.

#### 6.3.2. Categorical encoding <a name="categoricalEncoding"></a>

When preprocessing data for the machine learning model, we had dropped categorical data that may prove useful to improve the model's performance.

We can encode the data from the 'Status' column which contains categorical data about a country's 'Developed' or 'Developing' status.

This would result in an additional column that can perhaps be useful in enhancing the model.
"""

# Create a copy of the last DataFrame that contains 'Status' data
life_exp_df12 = life_exp_df9.copy()

# Adapted from https://stackoverflow.com/a/70537483
# Add a new column that encodes Status column data ()'Developed' to 1 and 'Developing' to 0)
life_exp_df12['status_numeric'] = life_exp_df9.apply(
    lambda row: 1.0 if 'Developed' == row['status'] else 0.0,
    axis=1
)

# Create a copy of the DataFrame used for the machine learning model
temp_df = life_exp_df11.copy()

# Add a new column to the temporary DataFrame to hold the encoded 'Status' data
temp_df['status_numeric'] = life_exp_df12['status_numeric']

# Overwrite the life_exp_df12 DataFrame with the data from the temporary DataFrame
life_exp_df12 = temp_df

# Printing the first 5 rows
life_exp_df12.head()

"""We can check if adding this column improves the initial linear regression model's score.

Arranging the new features matrix and target variable,
"""

X_life_exp1 = life_exp_df12.drop('life_expectancy', axis=1) # features matrix
y_life_exp1 = life_exp_df12['life_expectancy'] # target variable

# Leaving a third of the data out to test the model
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_life_exp1, y_life_exp1, test_size=1/3, random_state=42)

# Use the training datasets to fit the initial linear regression model
linear_reg_model.fit(X_train1, y_train1)

# Calculating model score
lin_reg_cat_score = linear_reg_model.score(X_test1, y_test1)
lin_reg_cat_score
print(f"Linear regression with categorical encoding score (r2): {lin_reg_cat_score:.5f}")

"""# 7. Support vector machine<a name="svr"></a>

Using support vector regression,
"""

from sklearn.svm import SVR

# Initialize SVR model
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# Fit the model to the training data
svr_model.fit(X_train, y_train)

# Apply model on unseen data
svr_score = svr_model.score(X_test, y_test)
print(f"SVR score (r2): {svr_score:.5f}")

# Model evaluation
# Predict on the test set
y_pred5 = svr_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse_sv_reg = mean_squared_error(y_test, y_pred5)
# Calculate mean absolute error value
mae_sv_reg = mean_absolute_error(y_test, y_pred5)

print(f"SVR Mean Squared Error: {mse_sv_reg:.5f}")
print(f"SVR Mean Absolute Error: {mae_sv_reg:.5f}")

"""# 8. K-Nearest Neighbor <a name="knn"></a>"""

from sklearn.neighbors import KNeighborsRegressor

# Initialize KNR model
knr_model = KNeighborsRegressor()

# Fit the model to the training data
knr_model.fit(X_train, y_train)

# Apply model on unseen data
knr_score = knr_model.score(X_test, y_test)
print(f"KNR score (r2): {knr_score:.5f}")

# Model evaluation
# Predict on the test set
y_pred6 = knr_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse_knr = mean_squared_error(y_test, y_pred6)
# Calculate mean absolute error value
mae_knr = mean_absolute_error(y_test, y_pred6)

print(f"KNR Mean Squared Error: {mse_knr:.5f}")
print(f"KNR Mean Absolute Error: {mae_knr:.5f}")

"""# 9. Tree based models and ensembles <a name="treeModels"></a>

### 9.1. Decision tree <a name="decisionTree"></a>
"""

from sklearn.tree import DecisionTreeRegressor

# Initialize decision tree regressor model
dt_reg_model = DecisionTreeRegressor(random_state=42)

# Fit the model to the training data
dt_reg_model.fit(X_train, y_train)

# Apply model on unseen data
dt_reg_score = dt_reg_model.score(X_test, y_test)
print(f"Decision Tree reg score (r2): {dt_reg_score:.5f}")

# Model evaluation
# Predict on the test set
y_pred7 = dt_reg_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse_dt_reg = mean_squared_error(y_test, y_pred7)
# Calculate mean absolute error value
mae_dt_reg = mean_absolute_error(y_test, y_pred7)

print(f"Decision Tree Mean Squared Error: {mse_dt_reg:.5f}")
print(f"Decision Tree Mean Absolute Error: {mae_dt_reg:.5f}")

"""Visualize the top of decision tree to see the most important features,"""

from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(dt_reg_model, max_depth=2, out_file=None,
                           feature_names=X_life_exp.columns,
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)

graph

"""We can create a scatter plot at each of the nodes of the tree using the dtreeviz package."""

dtr = DecisionTreeRegressor(random_state=42, max_depth=2)
dtr.fit(X_train, y_train)

viz_model = dtreeviz.model(dtr, X_life_exp, y_life_exp, target_name='Life expectancy ', feature_names=X_life_exp.columns)

viz_model.view()

"""### 9.2. Random forest <a name="randomForest"></a>"""

from sklearn.ensemble import RandomForestRegressor

# Initialize random forest regressor model
rf_reg_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf_reg_model.fit(X_train, y_train)

# Apply model on unseen data
rf_reg_score = rf_reg_model.score(X_test, y_test)
print(f"Random forest reg score (r2): {rf_reg_score:.5f}")

# Model evaluation
# Predict on the test set
y_pred8 = rf_reg_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse_rf_reg = mean_squared_error(y_test, y_pred8)
# Calculate mean absolute error value
mae_rf_reg = mean_absolute_error(y_test, y_pred8)

print(f"Random Forest Mean Squared Error: {mse_rf_reg:.5f}")
print(f"Random Forest Mean Absolute Error: {mae_rf_reg:.5f}")

"""We can create a histogram of residuals to check their normality,"""

fig, ax = plt.subplots(figsize=(6, 4))
resids = y_test - y_pred8
pd.Series(resids, name="resoduals").plot.hist(bins=20, ax=ax, title="Residual Histogram")

"""We can also create a probability plot of residuals,"""

from scipy import stats
fig, ax = plt.subplots(figsize=(6, 4))
stats.probplot(resids, plot=ax)
plt.show()

"""It can be seen that the samples plotted against the quantiles fail to completely line up, and so the residuals are not normal.

We can further use the Kolmogorov-Smirnov test to measure whether the distribution is normal.
"""

stats.kstest(resids, cdf="norm")

"""It can be seen that the p-value is significant (<0.05), and the residuals are not normal.

We can also use a prediction error plot to visualize the real targets against the predicted values.
"""

from yellowbrick.regressor import PredictionError

fig, ax = plt.subplots(figsize=(6, 6))
pred_err_viz = PredictionError(rf_reg_model)
pred_err_viz.fit(X_train, y_train)
pred_err_viz.score(X_test, y_test)
pred_err_viz.poof()

"""Looking at the plot, it can be seen that the model's performance could be somewhat improved. For a perfect model, the predicted values would match the actual values for every observation in the data, and there would be no outlier points.

### 9.3. Gradient boosting machine <a name="gbm"></a>
"""

from sklearn.ensemble import GradientBoostingRegressor

# Initialize GBM model
gb_reg_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Fit the model to the training data
gb_reg_model.fit(X_train, y_train)

# Apply model on unseen data
gb_reg_score = gb_reg_model.score(X_test, y_test)
print(f"GBM reg score (r2): {gb_reg_score:.5f}")

# Model evaluation
# Predict on the test set
y_pred9 = gb_reg_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse_gb_reg = mean_squared_error(y_test, y_pred9)
# Calculate mean absolute error value
mae_gb_reg = mean_absolute_error(y_test, y_pred9)

print(f"GBM Mean Squared Error: {mse_gb_reg:.5f}")
print(f"GBM Mean Absolute Error: {mae_gb_reg:.5f}")

"""### 9.4. XGBoost regression <a name="xgboost"></a>"""

import xgboost as xgb

# Initialize XGBR
xgbr_model = xgb.XGBRegressor(random_state=42)

# Fit the model to the training data
xgbr_model.fit(X_train, y_train)

# Apply model on unseen data
xgb_reg_score = xgbr_model.score(X_test, y_test)
print(f"XGB reg score (r2): {xgb_reg_score:.5f}")

# Model evaluation
# Predict on the test set
y_pred10 = xgbr_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse_xgb_reg = mean_squared_error(y_test, y_pred10)
# Calculate mean absolute error value
mae_xgb_reg = mean_absolute_error(y_test, y_pred10)

print(f"XGB Mean Squared Error: {mse_xgb_reg:.5f}")
print(f"XGB Mean Absolute Error: {mae_xgb_reg:.5f}")

"""Plotting feature importance (F score),"""

fig, ax = plt.subplots(figsize=(6, 4))
xgb.plot_importance(xgbr_model, ax=ax)

"""Using YellowBrick to plot feature importance shows the percent importance of the features."""

from yellowbrick.model_selection import FeatureImportances

fig, ax = plt.subplots(figsize=(6, 4))
feature_imp_viz = FeatureImportances(xgbr_model)
feature_imp_viz.fit(X_train, y_train)
feature_imp_viz.poof()

"""# 10. Feedforward neural network <a name="fnn"></a>

Using a multi-layer perceptron regressor with grid search
"""

from sklearn.neural_network import MLPRegressor

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100), (200, 200)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01]
}

# Create the MLP regressor
mlp_regressor = MLPRegressor(random_state=42, max_iter=1000)

# Instantiate the GridSearchCV object
grid_search = GridSearchCV(estimator=mlp_regressor, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
print(best_params)

# Get best model
mlp_reg_model = grid_search.best_estimator_

# Apply model on unseen data
mlp_reg_score = mlp_reg_model.score(X_test, y_test)
print(f"MLP reg score (r2): {svr_score:.5f}")

# Model evaluation
# Predict on the test set
y_pred12 = mlp_reg_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse_mlp_reg = mean_squared_error(y_test, y_pred12)
# Calculate mean absolute error value
mae_mlp_reg = mean_absolute_error(y_test, y_pred12)

print(f"MLP Mean Squared Error: {mse_mlp_reg:.5f}")
print(f"MLP Mean Absolute Error: {mae_mlp_reg:.5f}")