import pandas as pd
import datetime as dt
import os
import httpx
import re
import time
import json
from dotenv import load_dotenv

## Github Token
load_dotenv()
github_token = os.getenv("GITHUB_TOKEN")

## Load Training and Test Data
traind = pd.read_csv('CryptoPondData/dataset.csv')
testd = pd.read_csv('CryptoPondData/test.csv')

## Repo DF
repo_urls = pd.concat([traind['project_a'], traind['project_b'], testd['project_a'], testd['project_b']]).unique()
repo_df = df = pd.DataFrame({
	'ID': range(1, len(repo_urls) + 1),
	'Value': repo_urls
})

## Get Repository Info Function
def get_repo_stats(repository_id: str):
	repository_id = re.search(r'github\.com/([^/]+/[^/]+)', repository_id)
	repo_id = repository_id.group(1)
	api_url = f"https://api.github.com/repos/{repo_id}"
	headers = {"Accept": "application/vnd.github+json","Authorization": f"Bearer {github_token}","X-GitHub-Api-Version": "2022-11-28",}
	client = httpx.Client()
	response = client.get(api_url, headers=headers)
	time.sleep(1)
	return response.json()

output_dir = 'CryptoPondData/repostats'
os.makedirs(output_dir, exist_ok=True)
for _, row in repo_df.iterrows():
	file_name = os.path.join(output_dir, f"{row['ID']}.json")
	json_content = get_repo_stats(row['Value'])
	with open(file_name, 'w', encoding='utf-8') as json_file:
		json.dump(json_content, json_file, indent=4)
	print(str(row['ID']))

## Parse Repository Info Function
def get_repo_stats(filename: str):
	with open(filename, 'r', encoding='utf-8') as f:
		data = json.load(f)
	repostats_dft = pd.DataFrame([{
		'ID' : data.get("id"),
		'isPrivate' : data.get("private"),
		'Description' : data.get("description"),
		'isFork' : data.get("fork"),
		'Created' : data.get("created_at"),
		'Updated' : data.get("updated_at"),
		'Size' : data.get("size"),
		'StarCount' : data.get("stargazers_count"),
		'Language' : data.get("language"),
		'hasIssues' : data.get("has_issues"),
		'hasProjects' : data.get("has_issues"),
		'hasDownloads' : data.get("has_issues"),
		'hasWiki' : data.get("has_issues"),
		'hasPages' : data.get("has_issues"),
		'hasDiscussions' : data.get("has_issues"),
		'Forks' : data.get("forks_count"),
		'IssueCount' : data.get("open_issues_count")
	}])
	return repostats_dft

repostats_df = pd.DataFrame()
output_dir = 'CryptoPondData/repostats'
for _, row in repo_df.iterrows():
	file_name = os.path.join(output_dir, f"{row['ID']}.json")
	repostats_dft = get_repo_stats(file_name)
	repostats_df = pd.concat([repostats_df, repostats_dft], ignore_index=True)
	print(str(row['ID']))

repostats_df = pd.concat([repo_df, repostats_df], axis=1)
repostats_df.to_csv("CryptoPondData/repostats_df.csv",index=False)