import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

# 1. Data loading
zori_df = pd.read_csv('ZORI rent.csv', index_col=0)
zordi_df = pd.read_csv('ZORDI rental demand.csv', index_col=0)

# List of specified states and cities
cities = [
    'CA:San Francisco', 'CA:San Jose', 'CA:Ukiah', 'CA:Red Bluff', 'CA:Susanville',
    'NV:Carson City', 'NV:Gardnerville Ranchos',
    'TX:Midland', 'TX:Odessa', 'TX:Victoria', 'TX:Athens', 'TX:Nacogdoches', 'TX:Stephenville', 'TX:Brenham',
    'KS:Manhattan', 'KS:Hutchinson', 'KS:Pittsburg',
    'NY:New York', 'NY:Glens Falls', 'NY:Corning', 'NY:Oneonta', 'NY:Amsterdam',
    'GA:Dalton'
]

correlation_results = []

for city in cities:
    # 2. Data cleaning and processing
    # Keep only the data for the current city
    city_zori = zori_df[zori_df['RegionName'].str.contains(city.split(':')[1])]
    city_zordi = zordi_df[zordi_df['RegionName'].str.contains(city.split(':')[1])]

    # Convert date columns to rows (wide to long format), ignoring non-date columns
    city_zori_melted = city_zori.melt(id_vars=['RegionName'], var_name='Date', value_name='ZORI')
    city_zordi_melted = city_zordi.melt(id_vars=['RegionName'], var_name='Date', value_name='ZORDI')

    # Try to convert date strings to datetime format
    city_zori_melted['Date'] = pd.to_datetime(city_zori_melted['Date'], errors='coerce')
    city_zordi_melted['Date'] = pd.to_datetime(city_zordi_melted['Date'], errors='coerce')

    # Drop rows that cannot be converted to date
    city_zori_melted.dropna(subset=['Date'], inplace=True)
    city_zordi_melted.dropna(subset=['Date'], inplace=True)

    # Convert numeric columns to float to ensure correct format
    city_zori_melted['ZORI'] = pd.to_numeric(city_zori_melted['ZORI'], errors='coerce')
    city_zordi_melted['ZORDI'] = pd.to_numeric(city_zordi_melted['ZORDI'], errors='coerce')

    # Drop rows that cannot be converted to numeric
    city_zori_melted.dropna(subset=['ZORI'], inplace=True)
    city_zordi_melted.dropna(subset=['ZORDI'], inplace=True)

    # Merge two datasets based on city and date
    combined_df = pd.merge(city_zori_melted, city_zordi_melted, on=['RegionName', 'Date'], how='inner')
    combined_df.set_index('Date', inplace=True)

    # Check if there is enough data for analysis
    if len(combined_df) < 2:
        print(f"{city} has insufficient data for analysis.")
        continue

    # Filter the time range from 2020/6/30 to 2024/10/31
    combined_df = combined_df.loc['2020-06-30':'2024-10-31']

    # Check again if data is available
    if combined_df.empty:
        print(f"{city} has insufficient data for analysis.\n")
        continue

    # 3. Significant correlation analysis
    # Perform regression: Price(t) = β0 + β1 * Demand(t) + εt
    X = combined_df['ZORDI']
    y = combined_df['ZORI']
    X = sm.add_constant(X)  # Add constant term

    # Ensure data is in numeric array format
    X = np.asarray(X)
    y = np.asarray(y)

    model = sm.OLS(y, X).fit()
    coef = model.params[1]
    correlation_results.append((city, coef))
    
    if coef > 0:
        print(f"City with positive regression coefficient - {city}")
        print(model.summary())
   


# Plot bar chart comparing correlation coefficients across cities
cities, coefs = zip(*correlation_results)
plt.figure(figsize=(14, 7))
sns.barplot(x=list(cities), y=list(coefs))
plt.xticks(rotation=90)
plt.xlabel('city')
plt.ylabel('regression coefficient')
plt.title('Relevance of different cities')
plt.savefig('Relevance of different cities.png')
plt.show()


# In[26]:


# Combined cities analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

# 1. Data loading
zori_df = pd.read_csv('ZORI rent.csv', index_col=0)
zordi_df = pd.read_csv('ZORDI rental demand.csv', index_col=0)

# List of specified states and cities
cities = [
    'CA:San Francisco', 'CA:San Jose', 'CA:Ukiah', 'CA:Red Bluff', 'CA:Susanville',
    'NV:Carson City', 'NV:Gardnerville Ranchos',
    'TX:Midland', 'TX:Odessa', 'TX:Victoria', 'TX:Athens', 'TX:Nacogdoches', 'TX:Stephenville', 'TX:Brenham',
    'KS:Manhattan', 'KS:Hutchinson', 'KS:Pittsburg',
    'NY:New York', 'NY:Albany', 'NY:Glens Falls', 'NY:Corning', 'NY:Oneonta', 'NY:Amsterdam',
    'GA:Dalton', ]

correlation_results = []

# Combine data from all cities for analysis
combined_zori = pd.DataFrame()
combined_zordi = pd.DataFrame()

for city in cities:
    # 2. Data cleaning and processing
    # Keep only data for the current city
    city_zori = zori_df[zori_df['RegionName'].str.contains(city.split(':')[1])]
    city_zordi = zordi_df[zordi_df['RegionName'].str.contains(city.split(':')[1])]

    # Convert date columns to rows (wide to long format), ignoring non-date columns
    city_zori_melted = city_zori.melt(id_vars=['RegionName'], var_name='Date', value_name='ZORI')
    city_zordi_melted = city_zordi.melt(id_vars=['RegionName'], var_name='Date', value_name='ZORDI')

    # Try to convert date strings to datetime format
    city_zori_melted['Date'] = pd.to_datetime(city_zori_melted['Date'], errors='coerce')
    city_zordi_melted['Date'] = pd.to_datetime(city_zordi_melted['Date'], errors='coerce')

    # Drop rows that cannot be converted to date
    city_zori_melted.dropna(subset=['Date'], inplace=True)
    city_zordi_melted.dropna(subset=['Date'], inplace=True)

    # Convert numeric columns to float
    city_zori_melted['ZORI'] = pd.to_numeric(city_zori_melted['ZORI'], errors='coerce')
    city_zordi_melted['ZORDI'] = pd.to_numeric(city_zordi_melted['ZORDI'], errors='coerce')

    # Drop rows that cannot be converted to numeric
    city_zori_melted.dropna(subset=['ZORI'], inplace=True)
    city_zordi_melted.dropna(subset=['ZORDI'], inplace=True)

    # Merge two datasets based on city and date
    combined_df = pd.merge(city_zori_melted, city_zordi_melted, on=['RegionName', 'Date'], how='inner')
    combined_df.set_index('Date', inplace=True)

    # Filter time range from 2020/6/30 to 2024/10/31
    combined_df = combined_df.loc['2020-06-30':'2024-10-31']

    # Combine all city data
    combined_zori = pd.concat([combined_zori, combined_df['ZORI']])
    combined_zordi = pd.concat([combined_zordi, combined_df['ZORDI']])

# 3. Significant correlation analysis - combined city data
combined_data = pd.concat([combined_zori, combined_zordi], axis=1, keys=['ZORI', 'ZORDI']).dropna()

X = combined_data['ZORDI']
y = combined_data['ZORI']
X = sm.add_constant(X)  # Add constant term

# Ensure numeric array format
X = np.asarray(X)
y = np.asarray(y)

model = sm.OLS(y, X).fit()
print("Regression analysis for combined cities:")
print(model.summary())
 # Perform regression: Price(t) = β0 + β1 * Demand(t) + εt
X = combined_df['ZORDI']
y = combined_df['ZORI']
X = sm.add_constant(X)  # Add constant term

# Ensure numeric array format
X = np.asarray(X)
y = np.asarray(y)
model = sm.OLS(y, X).fit()
coef = model.params[1]
correlation_results.append((city, coef))