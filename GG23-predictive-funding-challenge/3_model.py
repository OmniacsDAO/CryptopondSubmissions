import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import shap


#############################################
## Predict Contribution
#############################################
trainfeature_df = pd.read_csv('~/Desktop/GitcoinCryptoPond/dataset/train.csv')
trainfeature_df = trainfeature_df[['Amt'] + [str(i) for i in range(1, 769)]]
testfeature_df = pd.read_csv('~/Desktop/GitcoinCryptoPond/dataset/test.csv')
testfeature_df = testfeature_df[[str(i) for i in range(1, 769)]]
testd = pd.read_csv('~/Desktop/GitcoinCryptoPond/dataset/projects_Apr_1.csv')
testd['MPOOL'] = testd['ROUND'].apply(lambda x: 600000 if x == 'MATURE BUILDERS' else 200000)
X = trainfeature_df.iloc[:, 1:]
y = trainfeature_df.Amt
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.075, random_state=12)
regr = XGBRegressor(objective ='reg:squarederror',eval_metric = 'rmse', n_estimators = 500, max_depth=6,learning_rate=0.05, seed = 12, subsample = 1, colsample_bytree=.5)
regr.fit(X_train, y_train)
print(mean_squared_error(y_test, regr.predict(X_test)))
print(mean_squared_error(y_train, regr.predict(X_train)))
testd['CONTRIBUTION'] = regr.predict(testfeature_df)
#############################################
#############################################


#############################################
## Predict MatchingPool
#############################################
trainfeature_df = pd.read_csv('~/Desktop/GitcoinCryptoPond/dataset/train.csv')
trainfeature_df = trainfeature_df[['MatchingPoolPct'] + [str(i) for i in range(1, 769)]]
X = trainfeature_df.iloc[:, 1:]
y = trainfeature_df.MatchingPoolPct
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.075, random_state=12)
regr = XGBRegressor(objective ='reg:squarederror',eval_metric = 'rmse', n_estimators = 500, max_depth=6,learning_rate=0.05, seed = 12, subsample = 1, colsample_bytree=.5)
regr.fit(X_train, y_train)
print(mean_squared_error(y_test, regr.predict(X_test)))
print(mean_squared_error(y_train, regr.predict(X_train)))
testd['MPOOLPCT'] = regr.predict(testfeature_df)
#############################################
#############################################


#############################################
## Predict the sum
#############################################
testd.loc[testd['Live'] == 0, 'CONTRIBUTION'] = 0
testd.loc[testd['Live'] == 0, 'MPOOLPCT'] = 0
testd['MPOOLPCT1'] = testd.groupby('ROUND').apply(lambda g: (g['MPOOLPCT'] / g['MPOOLPCT'].sum()) * 1 * g['MPOOL']).reset_index(level=0, drop=True)
testd['MPOOLPCT12'] = testd.groupby('ROUND').apply(lambda g: (g['MPOOLPCT'] / g['MPOOLPCT'].sum()) * 1.2 * g['MPOOL']).reset_index(level=0, drop=True)
testd.to_csv('~/Desktop/GitcoinCryptoPond/dataset/pred/pred.csv',index=False)

submission1 = testd
submission1['AMOUNT'] = submission1['CONTRIBUTION']+submission1['MPOOLPCT1']
submission1 = submission1.iloc[:, :4]
submission1.columns.values[0] = 'PROJECT_ID'
submission1.to_csv('~/Desktop/GitcoinCryptoPond/dataset/pred/submission1.csv',index=False)

submission2 = testd
submission2['AMOUNT'] = submission2['MPOOLPCT12']
submission2 = submission2.iloc[:, :4]
submission2.columns.values[0] = 'PROJECT_ID'
submission2.to_csv('~/Desktop/GitcoinCryptoPond/dataset/pred/submission2.csv',index=False)

submission3 = testd
submission3.loc[submission3['ROUND'] == "MATURE BUILDERS", 'CONTRIBUTION'] = 0
submission3['AMOUNT'] = submission3['CONTRIBUTION']+submission3['MPOOLPCT1']
submission3 = submission3.iloc[:, :4]
submission3.columns.values[0] = 'PROJECT_ID'
submission3.to_csv('~/Desktop/GitcoinCryptoPond/dataset/pred/submission3.csv',index=False)

#############################################
#############################################

