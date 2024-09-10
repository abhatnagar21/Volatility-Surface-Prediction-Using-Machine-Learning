import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Generate synthetic option data
np.random.seed(42)
data_size = 1000

# Features: Strike Price, Time to Maturity, and Option Price
data = pd.DataFrame({
    'Strike Price': np.random.uniform(80, 120, data_size),
    'Time to Maturity': np.random.uniform(0.1, 2, data_size),  # in years
    'Option Price': np.random.uniform(5, 15, data_size)
})

# Implied Volatility Calculation (Simplified Black-Scholes)
def implied_volatility(S, K, T, r, C):
    from scipy.optimize import minimize
    from scipy.stats import norm

    def bs_price(sigma):
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def loss(sigma):
        return (bs_price(sigma) - C) ** 2

    result = minimize(loss, 0.2, bounds=[(0.01, 2)])
    return result.x[0]

data['Implied Volatility'] = data.apply(lambda row: implied_volatility(row['Strike Price'], row['Strike Price']*0.95, row['Time to Maturity'], 0.01, row['Option Price']), axis=1)

# Add additional features
data['Moneyness'] = data['Option Price'] / data['Strike Price']
data['Log_Moneyness'] = np.log(data['Moneyness'])

# Prepare features and target variable
X = data[['Strike Price', 'Time to Maturity', 'Moneyness', 'Log_Moneyness']]
y = data['Implied Volatility']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Cross-validation score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation Mean Squared Error: {-np.mean(cv_scores)}")

# Plot the predicted vs actual volatility
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Predictions', color='b')
plt.xlabel('Actual Implied Volatility')
plt.ylabel('Predicted Implied Volatility')
plt.title('Actual vs Predicted Implied Volatility')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
plt.legend()
plt.show()

# Heatmap of Actual vs Predicted Implied Volatility
plt.figure(figsize=(12, 6))
heatmap_data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
heatmap_data['Error'] = heatmap_data['Predicted'] - heatmap_data['Actual']
sns.heatmap(heatmap_data.pivot_table(index='Actual', columns='Predicted', values='Error'), cmap='coolwarm', annot=False)
plt.title('Heatmap of Prediction Errors')
plt.xlabel('Predicted Implied Volatility')
plt.ylabel('Actual Implied Volatility')
plt.show()

# Feature Importance
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(12, 6))
sns.barplot(x=features, y=importances)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Residual Plot
plt.figure(figsize=(12, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Predicted Implied Volatility')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Visualize the Volatility Surface
# Generate a grid of strike prices and maturities
strike_prices = np.linspace(X['Strike Price'].min(), X['Strike Price'].max(), 50)
maturities = np.linspace(X['Time to Maturity'].min(), X['Time to Maturity'].max(), 50)
strike_prices_grid, maturities_grid = np.meshgrid(strike_prices, maturities)
X_grid = np.c_[strike_prices_grid.ravel(), maturities_grid.ravel(), 
                np.random.uniform(0.5, 1.5, size=strike_prices_grid.ravel().shape[0]),  # Moneyness
                np.random.uniform(-0.1, 0.1, size=strike_prices_grid.ravel().shape[0])]  # Log_Moneyness

# Predict volatility over the grid
volatility_grid = model.predict(X_grid).reshape(strike_prices_grid.shape)

# Plot the 3D volatility surface
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(strike_prices_grid, maturities_grid, volatility_grid, cmap='viridis')

ax.set_xlabel('Strike Price')
ax.set_ylabel('Time to Maturity')
ax.set_zlabel('Implied Volatility')
ax.set_title('Implied Volatility Surface')

plt.show()
