
import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)
df1 = pd.read_csv(r'C:\Users\jense\OneDrive\Desktop\Meta数据分析\IG_market_dataset.csv')
df2 = pd.read_csv(r'C:\Users\jense\OneDrive\Desktop\Meta数据分析\ER.csv')
df3 = pd.read_csv(r'C:\Users\jense\OneDrive\Desktop\Meta数据分析\country_stats.csv')
# Convert strings with commas to floats
df3['area_km2'] = df3['area_km2'].str.replace(',', '').astype(float)

df3[['GDP','currency code']] = df3.gdp_local_currency_trillion.str.split(expand=True)
df3[['gdp_per_capital','currency code']] =df3.gdp_per_capita_local_currency.str.split(expand=True)
result = pd.merge(df3, df2, on='currency code', how='inner')
print(result[['GDP','gdp_per_capital','exchange_rate_to_usd']])
result['GDP'] = pd.to_numeric(result['GDP'], errors='coerce')
result['gdp_per_capital'] = pd.to_numeric(result['gdp_per_capital'], errors='coerce')
result['exchange_rate_to_usd'] = pd.to_numeric(result['exchange_rate_to_usd'], errors='coerce')

result['GDP_USD'] = result['GDP'] * result['exchange_rate_to_usd']
result['GDP_per_capita_USD'] = result['gdp_per_capital'] / result['exchange_rate_to_usd']

selected_columns = result[['country_region','GDP','gdp_per_capital','currency code','GDP_USD','GDP_per_capita_USD']]
print(selected_columns)
#Top5 GDP Countries
selected_columns_GDP = selected_columns.sort_values(by='GDP_USD', ascending=False)
#Top 5 hdi county order by
selected_columns_HDI = result.sort_values(by='hdi', ascending=False)

print(selected_columns)
print("Ranked HDI\n", selected_columns_HDI.head(5))

avg_gdp_per_capita = result['GDP_per_capita_USD'].mean()

# Filtering countries with GDP_per_capita less than the average
below_avg = result[result['GDP_per_capita_USD'] < avg_gdp_per_capita]

# Filtering countries with GDP_per_capita greater than the average
above_avg = result[result['GDP_per_capita_USD'] > avg_gdp_per_capita]

# If you want them in a single DataFrame, you can concatenate them:
filtered_df = pd.concat([below_avg, above_avg])
print("below GDP per capita average\n", below_avg)
print("above GDP per capita average\n", above_avg)

new_table= pd.merge(df1, result, on='country_region', how='inner')

#CC On Margin
correlation1 = new_table['margin_bn_usd'].corr(new_table['population_million'])
print("the correlation between margin in usd and population in million is:", correlation1)

correlation2 = new_table['margin_bn_usd'].corr(new_table['area_km2'])
print("the correlation between margin in usd and area is:", correlation2)

correlation3 = new_table['margin_bn_usd'].corr(new_table['GDP_USD'])
print("the correlation between margin in usd and GDP is:", correlation3)

correlation4 = new_table['margin_bn_usd'].corr(new_table['GDP_per_capita_USD'])
print("the correlation between margin in usd and GDP per capita is:", correlation4)

correlation5 = new_table['margin_bn_usd'].corr(new_table['hdi'])
print("the correlation between margin in usd and hdi is:", correlation5)

correlation6 = new_table['margin_bn_usd'].corr(new_table['population_density_km2'])
print("the correlation between margin in usd and population_density_km2 is:", correlation6)

# CC on revenue
correlation7 = new_table['revenue_bn_usd'].corr(new_table['population_million'])
print("the correlation between margin in usd and population in million is:", correlation7)

correlation8 = new_table['revenue_bn_usd'].corr(new_table['area_km2'])
print("the correlation between total revenue in usd and area is:", correlation8)

correlation9 = new_table['revenue_bn_usd'].corr(new_table['GDP_USD'])
print("the correlation between total revenue in usd and GDP is:", correlation9)

correlation10 = new_table['revenue_bn_usd'].corr(new_table['GDP_per_capita_USD'])
print("the correlation between total revenuein usd and GDP per capita is:", correlation10)

correlation11 = new_table['revenue_bn_usd'].corr(new_table['hdi'])
print("the correlation between total revenue in usd and hdi is:", correlation11)

correlation12 = new_table['revenue_bn_usd'].corr(new_table['population_density_km2'])
print("the correlation between total revenue in usd and population_density_km2 is:", correlation12)

#CC on Users
correlation13 = new_table['users_million'].corr(new_table['population_million'])
print("the correlation between users and population in million is:", correlation13)

correlation14 = new_table['users_million'].corr(new_table['area_km2'])
print("the correlation between users and area is:", correlation14)

correlation15 = new_table['users_million'].corr(new_table['GDP_USD'])
print("the correlation between users and GDP is:", correlation15)

correlation16 = new_table['users_million'].corr(new_table['GDP_per_capita_USD'])
print("the correlation between users  and GDP per capita is:", correlation16)

correlation17 = new_table['users_million'].corr(new_table['hdi'])
print("the correlation between users  and hdi is:", correlation17)

correlation18 = new_table['users_million'].corr(new_table['population_density_km2'])
print("the correlation between users and population_density_km2 is:", correlation18)









