library(httr)
library(jsonlite)
library(readr)
library(ollamar)

source("~/Desktop/GitcoinCryptoPond/0_graphql_queries.R")

## Prepare Training Data
datap <- read_csv("~/Desktop/GitcoinCryptoPond/dataset/manualmatches2.csv",show_col_types = FALSE)
datap <- data.frame(
					Round = datap$ROUND,
					ProjID = datap$PROJECT_ID,
					description = datap$Description
			)

## Add Description Embeddings
uniq_ids <- na.omit(unique(datap$ProjID))
for (idxproj in 1:length(uniq_ids))
{
  tdata <- get_gitcoin_project(uniq_ids[idxproj])$metadata$description  
  if (!is.null(tdata))	datap$description[which(datap$ProjID == uniq_ids[idxproj])] <- tdata
  message(idxproj)
}
datap <- cbind(datap,t(rep(NA,768)))
for(idx in 1:nrow(datap))
{
	if(!is.na(datap$description[idx]))
	{
		datap[idx,4:771] <- embeddings(model = "nomic-embed-text:v1.5",prompt = datap$description[idx],host="http://192.168.11.98:9000")
	}
	message(idx)
}
write_csv(datap,"~/Desktop/GitcoinCryptoPond/dataset/test.csv")
