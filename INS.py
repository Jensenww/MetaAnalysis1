import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
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

result['GDP_USD'] = (result['GDP'] / result['exchange_rate_to_usd'])
result['GDP_per_capita_USD'] = (result['gdp_per_capital'] / result['exchange_rate_to_usd'])


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

X = new_table[['GDP_USD', 'GDP_per_capita_USD', 'hdi', 'population_density_km2','area_km2','population_million']]
y = new_table[['margin_bn_usd']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
print (linreg.intercept_)


coef_list = [float(coef) for coef in linreg.coef_.flatten()]
formatted_coef_list = ['{:.10f}'.format(coef) for coef in coef_list]
print(formatted_coef_list)


#模型拟合测试集
y_pred = linreg.predict(X_test)
from sklearn import metrics
# 用scikit-learn计算MSE

print ("MSE:",metrics.mean_squared_error(y_test, y_pred))
# 用scikit-learn计算RMSE
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming you have already generated X and y using make_regression
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Initialize the scaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit on the training data and transform both training and test data
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Fit on the training targets and transform both training and test targets
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()


# Fit the regressor on the scaled data
regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train_scaled, y_train_scaled)

# Make predictions on the scaled test data
predictions_scaled = regr.predict(X_test_scaled)

#  original scale, you can inverse transform the predictions
predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()


# Print the results
print(predictions_scaled)
print(predictions)
print(regr.score(X_test_scaled, y_test_scaled))

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"RMSE: {mse** 0.5}")

# compute the residuals:
#predictions = regr.predict(X_test_scaled)
#residuals = y_test_scaled - predictions

# 1. Residuals vs. Fitted values plot
#plt.figure(figsize=(8, 6))
#plt.scatter(predictions, residuals)
#plt.axhline(y=0, color='r', linestyle='--')
#plt.xlabel("Fitted values")
#plt.ylabel("Residuals")
#plt.title("Residuals vs. Fitted Values")
#plt.show()

# 2. Histogram of Residuals
#plt.figure(figsize=(8, 6))
#plt.hist(residuals, bins=30, edgecolor='k')
#plt.xlabel("Residual")
#plt.ylabel("Frequency")
#plt.title("Histogram of Residuals")
#plt.show()

#Prediction On China Economic Data
df4 = pd.read_csv(r'C:\Users\jense\OneDrive\Desktop\Meta数据分析\ChinaEconomicData.csv')

# Assuming new_data is your new dataset
new_X = df4[['GDP_USD', 'GDP_per_capita_USD', 'hdi', 'population_density_km2', 'area_km2', 'population_million']]

# Scale the new data using the same scaler object you used for the training data
new_X_scaled = scaler_X.transform(new_X)

# Use the trained model to make predictions
new_predictions_scaled = regr.predict(new_X_scaled)

# Convert the scaled predictions back to original scale if needed
new_predictions = scaler_y.inverse_transform(new_predictions_scaled.reshape(-1, 1)).ravel()

print("China's margin is:",new_predictions)


