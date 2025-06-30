import pandas as pd
import math
from datetime import datetime

## Load Training and Test Data
traind = pd.read_csv('CryptoPondData/dataset.csv')
testd = pd.read_csv('CryptoPondData/test.csv')
repoemb_df = pd.read_csv("CryptoPondData/repoemb_df.csv")
repostats_df = pd.read_csv("CryptoPondData/repostats_df.csv")
repostatsoso_df = pd.read_csv("CryptoPondData/repostatsoso_df.csv")

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
	stats_df['TotalAmount'] = row.total_amount_usd
	stats_df['TotalAmountLog'] = math.log(row.total_amount_usd+1)
	funder_df = pd.DataFrame([(traind.funder.unique()==row.funder).astype(int)],columns=traind.funder.unique())
	quarter_df = pd.DataFrame([(traind.quarter.unique()==row.quarter).astype(int)],columns=traind.quarter.unique())
	final_df = pd.concat([stats_df, a_embrow.add_prefix('A_'), b_embrow.add_prefix('B_'),funder_df,quarter_df], axis=1)
	if type=="train":
		final_df['Y'] = row.weight_a
	return(final_df)

## Define function to derive features
def get_repo_featuesB(row,type):
	a_embrow = (repoemb_df.loc[repoemb_df['Value'] == row.project_b].iloc[:,2:]).reset_index()
	a_stats = (repostats_df.loc[repostats_df['Value'] == row.project_b]).reset_index()
	a_statsoso = (repostatsoso_df.loc[repostatsoso_df['repo_url'] == row.project_b]).reset_index()
	b_embrow = (repoemb_df.loc[repoemb_df['Value'] == row.project_a].iloc[:,2:]).reset_index()
	b_stats = (repostats_df.loc[repostats_df['Value'] == row.project_a]).reset_index()
	b_statsoso = (repostatsoso_df.loc[repostatsoso_df['repo_url'] == row.project_a]).reset_index()
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
	stats_df['TotalAmount'] = row.total_amount_usd
	stats_df['TotalAmountLog'] = math.log(row.total_amount_usd+1)
	funder_df = pd.DataFrame([(traind.funder.unique()==row.funder).astype(int)],columns=traind.funder.unique())
	quarter_df = pd.DataFrame([(traind.quarter.unique()==row.quarter).astype(int)],columns=traind.quarter.unique())
	final_df = pd.concat([stats_df, a_embrow.add_prefix('A_'), b_embrow.add_prefix('B_'),funder_df,quarter_df], axis=1)
	if type=="train":
		final_df['Y'] = row.weight_b
	return(final_df)

## Create Features Data Frame
trainfeature_df = pd.DataFrame()
for _, row in traind.iterrows():
	feature_dftA = get_repo_featuesA(row,type="train")
	feature_dftB = get_repo_featuesB(row,type="train")
	trainfeature_df = pd.concat([trainfeature_df, feature_dftA, feature_dftB], ignore_index=True)

testfeature_df = pd.DataFrame()
for _, row in testd.iterrows():
	feature_dftA = get_repo_featuesA(row,type="test")
	testfeature_df = pd.concat([testfeature_df, feature_dftA], ignore_index=True)

trainfeature_df.to_csv("CryptoPondData/trainfeatures.csv",index=False)
testfeature_df.to_csv("CryptoPondData/testfeatures.csv",index=False)

