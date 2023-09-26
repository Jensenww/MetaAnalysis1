
import pandas as pd
df1 = pd.read_csv(r'C:\Users\jense\OneDrive\Desktop\Meta数据分析\IG_market_dataset.csv')
df2 = pd.read_csv(r'C:\Users\jense\OneDrive\Desktop\Meta数据分析\ER.csv')
df3 = pd.read_csv(r'C:\Users\jense\OneDrive\Desktop\Meta数据分析\country_stats.csv')

df3[['GDP','currency code']] = df3.gdp_local_currency_trillion.str.split(expand=True)
df3[['gdp_per_capital','currency code']] =df3.gdp_per_capita_local_currency.str.split(expand=True)
result = pd.merge(df3, df2, on='currency code', how='inner')
print(result[['GDP','gdp_per_capital','exchange_rate_to_usd']])
result['GDP'] = pd.to_numeric(result['GDP'], errors='coerce')
result['gdp_per_capital'] = pd.to_numeric(result['gdp_per_capital'], errors='coerce')
result['exchange_rate_to_usd'] = pd.to_numeric(result['exchange_rate_to_usd'], errors='coerce')

result['GDP_USD'] = result['GDP'] * result['exchange_rate_to_usd']
result['GDP_per_capita_USD'] = result['gdp_per_capital'] * result['exchange_rate_to_usd']

selected_columns = result[['country_region','GDP','gdp_per_capital','currency code','GDP_USD','GDP_per_capita_USD']]
print(selected_columns)
#Top5 GDP Countries
selected_columns_GDP = selected_columns.sort_values(by='GDP_USD', ascending=False)
#Top 5 hdi county order by
selected_columns_HDI = result.sort_values(by='hdi', ascending=False)

print(selected_columns)
print(selected_columns_HDI)

for country in result['country_region']:
    if (result['GDP_per_capita_USD'] < result['GDP_per_capita_USD'].mean()).all():
        print("Countries had less than average GDP per Capita are", country)
    else :
        (result['GDP_per_capita_USD'] > result['GDP_per_capita_USD'].mean()).all()
        print("Countries had more than average GDP per Capita are", country)




