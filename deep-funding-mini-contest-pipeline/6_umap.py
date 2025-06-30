import pandas as pd
from itertools import combinations
import math
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

## Load Training and Test Data
traind = pd.read_csv('CryptoPondData/dataset.csv')
testd = pd.read_csv('CryptoPondData/test.csv')
repoemb_df = pd.read_csv("CryptoPondData/repoemb_df.csv")
repostats_df = pd.read_csv("CryptoPondData/repostats_df.csv")
repostatsoso_df = pd.read_csv("CryptoPondData/repostatsoso_df.csv")
repo_urls = pd.concat([traind['project_a'], traind['project_b'], testd['project_a'], testd['project_b']]).unique()

## Pairwise combinations
def pairwise_combinations_df(lst):
    pairs = list(combinations(lst, 2))
    df = pd.DataFrame(pairs, columns=['project_a', 'project_b'])
    return df

alld = pairwise_combinations_df(repo_urls)

## Add Features to the pairs
## Define function to derive features
def get_repo_featuesA(row,type):
	a_embrow = (repoemb_df.loc[repoemb_df['Value'] == row.project_a].iloc[:,2:]).reset_index()
	a_stats = (repostats_df.loc[repostats_df['Value'] == row.project_a]).reset_index()
	a_statsoso = (repostatsoso_df.loc[repostatsoso_df['repo_url'] == row.project_a]).reset_index()
	b_embrow = (repoemb_df.loc[repoemb_df['Value'] == row.project_b].iloc[:,2:]).reset_index()
	b_stats = (repostats_df.loc[repostats_df['Value'] == row.project_b]).reset_index()
	b_statsoso = (repostatsoso_df.loc[repostatsoso_df['repo_url'] == row.project_b]).reset_index()
	stats_df = pd.DataFrame([{
								'A_isPrivate': int(a_stats.isPrivate[0]),
								'A_isFork': int(a_stats.isFork[0]),
								'A_Size': a_stats.Size[0],
								'A_SizeLog': math.log(a_stats.Size[0]+1),
								'A_Stars': a_stats.StarCount[0],
								'A_StarsLog': math.log(a_stats.StarCount[0]+1),
								'A_Forks': a_stats.Forks[0],
								'A_ForksLog': math.log(a_stats.Forks[0]+1),
								'A_Issues': a_stats.IssueCount[0],
								'A_hasIssues': int(a_stats.hasIssues[0]),
								'A_hasProjects': int(a_stats.hasProjects[0]),
								'A_hasDownloads': int(a_stats.hasDownloads[0]),
								'A_hasWiki': int(a_stats.hasWiki[0]),
								'A_hasPages': int(a_stats.hasPages[0]),
								'A_hasDiscussions': int(a_stats.hasDiscussions[0]),
								'A_NumPackages' : a_statsoso.num_packages[0],
								'A_NumDependents' : a_statsoso.num_dependents_in_oso[0],
								'A_CreatedAgeYears' : ((datetime.now()-datetime.strptime(a_stats.Created[0],'%Y-%m-%dT%H:%M:%SZ')).days+1)/365,
								'A_UpdatedAgeYears' : ((datetime.now()-datetime.strptime(a_stats.Updated[0],'%Y-%m-%dT%H:%M:%SZ')).days+1)/365,
								'A_CreatedAgeMonths' : ((datetime.now()-datetime.strptime(a_stats.Created[0],'%Y-%m-%dT%H:%M:%SZ')).days+1)/30,
								'A_UpdatedAgeMonths' : ((datetime.now()-datetime.strptime(a_stats.Updated[0],'%Y-%m-%dT%H:%M:%SZ')).days+1)/30,
								'B_isPrivate': int(b_stats.isPrivate[0]),
								'B_isFork': int(b_stats.isFork[0]),
								'B_Size': b_stats.Size[0],
								'B_SizeLog': math.log(b_stats.Size[0]+1),
								'B_Stars': b_stats.StarCount[0],
								'B_StarsLog': math.log(b_stats.StarCount[0]+1),
								'B_Forks': b_stats.Forks[0],
								'B_ForksLog': math.log(b_stats.Forks[0]+1),
								'B_Issues': b_stats.IssueCount[0],
								'B_hasIssues': int(b_stats.hasIssues[0]),
								'B_hasProjects': int(b_stats.hasProjects[0]),
								'B_hasDownloads': int(b_stats.hasDownloads[0]),
								'B_hasWiki': int(b_stats.hasWiki[0]),
								'B_hasPages': int(b_stats.hasPages[0]),
								'B_hasDiscussions': int(b_stats.hasDiscussions[0]),
								'B_NumPackages' : b_statsoso.num_packages[0],
								'B_NumDependents' : b_statsoso.num_dependents_in_oso[0],
								'B_CreatedAgeYears' : (datetime.now()-datetime.strptime(b_stats.Created[0],'%Y-%m-%dT%H:%M:%SZ')).days/365,
								'B_UpdatedAgeYears' : (datetime.now()-datetime.strptime(b_stats.Updated[0],'%Y-%m-%dT%H:%M:%SZ')).days/365,
								'B_CreatedAgeMonths' : (datetime.now()-datetime.strptime(b_stats.Created[0],'%Y-%m-%dT%H:%M:%SZ')).days/30,
								'B_UpdatedAgeMonths' : (datetime.now()-datetime.strptime(b_stats.Updated[0],'%Y-%m-%dT%H:%M:%SZ')).days/30
				}])
	stats_df['SizeRatio'] = stats_df.A_Size / stats_df.B_Size
	stats_df['SizeRatioLog'] = stats_df.A_SizeLog / stats_df.B_SizeLog
	stats_df['StarsRatio'] = stats_df.A_Stars / stats_df.B_Stars
	stats_df['StarsRatioLog'] = stats_df.A_StarsLog / stats_df.B_StarsLog
	stats_df['ForksRatio'] = stats_df.A_Forks / stats_df.B_Forks
	stats_df['ForksRatioLog'] = stats_df.A_ForksLog / stats_df.B_ForksLog
	stats_df['IssuesRatio'] = (stats_df.A_Issues+1) / (stats_df.B_Issues+1)
	stats_df['CreatedAgeRatioYears'] = stats_df.A_CreatedAgeYears/ stats_df.B_CreatedAgeYears
	stats_df['CreatedAgeRatioMonths'] = stats_df.A_CreatedAgeMonths/ stats_df.B_CreatedAgeMonths
	funder_df = pd.DataFrame([(traind.funder.unique()==row.funder).astype(int)],columns=traind.funder.unique())
	quarter_df = pd.DataFrame([(traind.quarter.unique()==row.quarter).astype(int)],columns=traind.quarter.unique())
	final_df = pd.concat([stats_df, a_embrow.add_prefix('A_'), b_embrow.add_prefix('B_'),funder_df,quarter_df], axis=1)
	if type=="train":
		final_df['Y'] = row.weight_a
	return(final_df)

## Append Features
allfeature_df_list = []
for _, row in alld.iterrows():
	feature_dftA = get_repo_featuesA(row,type="test")
	allfeature_df_list.append(feature_dftA)

allfeature_df = pd.concat(allfeature_df_list, ignore_index=True)

## Predict weights
trainfeature_df = pd.read_csv('CryptoPondData/trainfeatures.csv')
X = trainfeature_df.iloc[:, :-1]
y = trainfeature_df.Y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.075, random_state=12)
regr = XGBRegressor(objective ='reg:squarederror',eval_metric = 'rmse', n_estimators = 500, max_depth=6,learning_rate=0.1, seed = 12, subsample = 1, colsample_bytree=.7)
regr.fit(X_train, y_train)
allfeature_df['weight_a'] = regr.predict(allfeature_df)
allfeature_df['weight_b'] = 1 - allfeature_df.weight_a
allfeature_df = pd.concat([alld, allfeature_df], axis=1)

# Summing the weights based on project occurrences
project_weights = {}
for _, row in allfeature_df.iterrows():
    if row['project_a'] not in project_weights:
        project_weights[row['project_a']] = 0
    if row['project_b'] not in project_weights:
        project_weights[row['project_b']] = 0
    project_weights[row['project_a']] += row['weight_a']
    project_weights[row['project_b']] += row['weight_b']

repoemb_df['SumWeights'] = repoemb_df['Value'].map(project_weights)
repoemb_df.to_csv("HuggingFaceData/UMAPData.csv")


