import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load Data
trainfeature_df = pd.read_csv('CryptoPondData/trainfeatures.csv')
testfeature_df = pd.read_csv('CryptoPondData/testfeatures.csv')
testd = pd.read_csv('CryptoPondData/test.csv')

# Train test Split
X = trainfeature_df.iloc[:, :-1]
y = trainfeature_df.Y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.075, random_state=12)

# Define base models
xgb_model = XGBRegressor(objective='reg:squarederror', eval_metric='rmse', n_estimators=500, max_depth=6, learning_rate=0.1, subsample=1, colsample_bytree=0.7, random_state=12)
rf_model = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=12)
gb_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=12)
svr_model = SVR(kernel='rbf', C=100, gamma=0.01)

# Define stacking regressor
stacked_model = StackingRegressor(
    estimators=[('xgb', xgb_model), ('rf', rf_model),  ('gb', gb_model), ('svr', svr_model)],
    final_estimator=XGBRegressor(objective='reg:squarederror', eval_metric='rmse', n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=12),
    passthrough=True
)

# Fit the stacked model
stacked_model.fit(X_train, y_train)

# Evaluate the model
train_pred = stacked_model.predict(X_train)
test_pred = stacked_model.predict(X_test)

print(f"Train MSE: {mean_squared_error(y_train, train_pred)}")
print(f"Test MSE: {mean_squared_error(y_test, test_pred)}")

# Make predictions on test data
submission = testd[['id']]
submission['pred'] = stacked_model.predict(testfeature_df)

# Save submission
submission.to_csv('CryptoPondData/sub_stacked.csv', index=False)
