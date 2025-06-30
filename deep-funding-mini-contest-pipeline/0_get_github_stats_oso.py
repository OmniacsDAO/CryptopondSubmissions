import pandas as pd
import datetime as dt
import os
from google.cloud import bigquery

## Load Training and Test Data
traind = pd.read_csv('CryptoPondData/dataset.csv')
testd = pd.read_csv('CryptoPondData/test.csv')

## Generate Bigquery
repo_urls = pd.concat([traind['project_a'], traind['project_b'], testd['project_a'], testd['project_b']]).unique()

def stringify_array(arr):
    return "'" + "','".join(arr) + "'"

query = f"""
WITH repos AS (
  SELECT *
  FROM `oso_production.repositories_v0`
),
package_owners AS (
  SELECT
    package_owner_artifact_id,
    package_artifact_source,
    package_artifact_name,
    CONCAT(package_artifact_source, '/', package_artifact_name) AS package_tag
  FROM `oso_production.package_owners_v0`
  WHERE package_owner_artifact_id IN (SELECT artifact_id FROM repos)
),
oso_dependents AS (
  SELECT
    package_owners.package_owner_artifact_id,
    COUNT(DISTINCT package_owners.package_tag) AS num_packages,
    COUNT(DISTINCT sboms.from_artifact_namespace) AS num_dependents_in_oso,
    ARRAY_AGG(DISTINCT package_owners.package_tag) AS list_of_packages,
    ARRAY_AGG(DISTINCT sboms.from_artifact_namespace) AS list_of_dependents_in_oso
  FROM `oso_production.sboms_v0` AS sboms
  JOIN package_owners
    ON sboms.to_package_artifact_name = package_owners.package_artifact_name
    AND sboms.to_package_artifact_source = package_owners.package_artifact_source
  GROUP BY 1
),
grants AS (
  SELECT
    funding.to_project_id AS project_id,
    ARRAY_AGG(DISTINCT projects.display_name) AS list_of_funders,
    SUM(funding.amount) AS total_funding_usd,
    SUM(CASE WHEN funding.time > '2023-01-01' THEN funding.amount ELSE 0 END) AS total_funding_usd_since_2023
  FROM `oso_production.oss_funding_v0` AS funding
  JOIN `oso_production.projects_v1` AS projects
    ON funding.from_project_id = projects.project_id
  WHERE funding.from_project_name IN ('gitcoin', 'octant-golemfoundation', 'opencollective', 'optimism')
  GROUP BY 1
),
combined AS (
  SELECT
    repos.artifact_url AS repo_url,
    repos.artifact_namespace AS maintainer,
    repos.language,
    repos.is_fork,
    DATE(repos.created_at) as created_at,
    DATE(repos.updated_at) as updated_at,    
    repos.star_count, 
    repos.fork_count,
    COALESCE(oso_dependents.num_packages, 0) AS num_packages,
    COALESCE(oso_dependents.num_dependents_in_oso, 0) AS num_dependents_in_oso,
    oso_dependents.list_of_dependents_in_oso,
    oso_dependents.list_of_packages,
    grants.list_of_funders,
    COALESCE(grants.total_funding_usd, 0) AS total_funding_usd,
    COALESCE(grants.total_funding_usd_since_2023, 0) AS total_funding_usd_since_2023
  FROM repos
  LEFT JOIN oso_dependents
    ON repos.artifact_id = oso_dependents.package_owner_artifact_id
  LEFT JOIN grants
    ON repos.project_id = grants.project_id
)
SELECT
  *,
  PERCENT_RANK() OVER (ORDER BY num_dependents_in_oso) AS oso_dependency_rank,
  COUNT(*) OVER (PARTITION BY language) AS num_repos_in_same_language,  
  PERCENT_RANK() OVER (PARTITION BY language ORDER BY num_dependents_in_oso) AS oso_dependency_rank_for_language
FROM combined
WHERE repo_url IN ({stringify_array(repo_urls)})

"""

## Make request and pull data
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'oso_gcp_credentials.json'
client = bigquery.Client(project='total-bliss-390613')
results = client.query(query)
df = results.to_dataframe()

## Save Repo Data
df.to_csv("CryptoPondData/repostatsoso_df.csv",index=False)