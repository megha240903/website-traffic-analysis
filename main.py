# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from fbprophet import Prophet

# Load your website traffic data
df = pd.read_csv('website_traffic_data.csv.xlsx')  # Replace with your CSV file

# Convert 'timestamp' column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set timestamp as index
df.set_index('timestamp', inplace=True)

# Resample to daily data
daily_data = df.resample('D').sum()

# -------------------------------
# 1. Data Visualization
# -------------------------------

# Plot Daily Visitors and Pageviews
plt.figure(figsize=(12, 6))
plt.plot(daily_data.index, daily_data['visitors'], label='Visitors')
plt.plot(daily_data.index, daily_data['page_views'], label='Page Views')
plt.title('Daily Visitors and Page Views')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Bounce Rate Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['bounce_rate'], bins=20, kde=True)
plt.title('Bounce Rate Distribution')
plt.xlabel('Bounce Rate')
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# 2. User Segmentation
# -------------------------------

# Segment users by device type
df['device_type'] = df['user_agent'].str.contains('Mobile', case=False, na=False)
mobile_users = df[df['device_type'] == True]
desktop_users = df[df['device_type'] == False]

print(f"Mobile Users: {len(mobile_users)}")
print(f"Desktop Users: {len(desktop_users)}")

# -------------------------------
# 3. Time Series Forecasting using Prophet
# -------------------------------

# Prepare data for Prophet
prophet_df = daily_data[['page_views']].reset_index()
prophet_df.columns = ['ds', 'y']

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=30)  # predict next 30 days
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title("Forecast of Page Views (Next 30 Days)")
plt.xlabel("Date")
plt.ylabel("Page Views")
plt.tight_layout()
plt.show()

# -------------------------------
# 4. Machine Learning - Linear Regression
# -------------------------------

# Feature Engineering Example (Dummy)
daily_data['dayofweek'] = daily_data.index.dayofweek
daily_data['month'] = daily_data.index.month
daily_data['is_weekend'] = daily_data['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# Define Features and Target
X = daily_data[['dayofweek', 'month', 'is_weekend']]
y = daily_data['page_views']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Linear Regression MSE: {mse:.2f}")

