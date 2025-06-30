import pandas as pd
import datetime as dt
import os
from getpass import getpass
import requests
from git import Repo
import glob
import shutil
from ollama import Client
import time
from dotenv import load_dotenv

## Github Token
load_dotenv()
github_token = os.getenv("GITHUB_TOKEN")

## Load Training and Test Data
traind = pd.read_csv('CryptoPondData/dataset.csv')
testd = pd.read_csv('CryptoPondData/test.csv')

## Repo List
repo_urls = pd.concat([traind['project_a'], traind['project_b'], testd['project_a'], testd['project_b']]).unique()
repo_df = df = pd.DataFrame({
	'ID': range(1, len(repo_urls) + 1),
	'Value': repo_urls
})

## Github API Access
headers = {'Accept': 'application/vnd.github.v3+json','Authorization': f'token {github_token}'}

## Cloning Repos and read all readme files
docs = {}
column_names = [f'Embedding_{i}' for i in range(768)]
embeddings_df = pd.DataFrame(columns=column_names)
client = Client(host='127.0.0.1:9000')
for _, row in repo_df.iterrows():
	folder_name = os.path.join("CryptoPondData/repos", str(row['ID']))
	if os.path.exists(folder_name):
		shutil.rmtree(folder_name)
	os.makedirs(folder_name, exist_ok=True)
	Repo.clone_from(row['Value'], folder_name, depth=1)
	md_files = glob.glob(os.path.join(folder_name, '*.md')) + glob.glob(os.path.join(folder_name, '*.markdown'))
	# md_files = glob.glob(os.path.join(folder_name, '**', '*.md'), recursive=True) + glob.glob(os.path.join(folder_name, '**', '*.markdown'), recursive=True)
	if len(md_files)==0:
		md_files = glob.glob(os.path.join(folder_name, '*.txt'))
	if md_files:
		content = ""
		for md_file in md_files:
			with open(md_file, 'r', encoding='utf-8') as f:
				content += f.read() + "\n\n"
		docs[row['ID']] = content
		response = client.embeddings(model='nomic-embed-text:v1.5',prompt=content,options={'num_ctx': 8192})
		embeddings_df.loc[len(embeddings_df)] = response["embedding"]
	time.sleep(10)
	print(str(row['ID'])+"/"+str(len(embeddings_df)))

repoemb_df = pd.concat([repo_df, embeddings_df], axis=1)
repoemb_df.to_csv("CryptoPondData/repoemb_df.csv",index=False)
