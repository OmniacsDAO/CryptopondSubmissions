library(httr)
library(jsonlite)
library(readr)
library(ollamar)

source("~/Desktop/GitcoinCryptoPond/0_graphql_queries.R")

## Prepare Training Data
datap <- read_csv("~/Desktop/GitcoinCryptoPond/dataset/GG Allocation Since GG18.csv",show_col_types = FALSE,locale = locale(encoding = "UTF-7"))
datap <- data.frame(
					Round = as.numeric(as.factor(datap$`Gitcoin Round Id`)),
					ProjID = datap$`Gitcoin Project Id`,
					NumContributors = datap$`# of Contributors`,
					Amt = datap$`Contribution Amount`,
					MatchingAmt = datap$`Matching Amount`,
					TotalAmt = datap$`Contribution Amount`+datap$`Matching Amount`
			)
datap$MatchingPool = sapply(datap$Round,function(x,y)sum(datap$MatchingAmt[datap$Round==x]),y=datap)
datap <- datap[datap$MatchingPool>0,]

## Add Embedings
datap$description <- NA
uniq_ids <- unique(datap$ProjID)
for (idxproj in 1:length(uniq_ids))
{
  tdata <- get_gitcoin_project(uniq_ids[idxproj])$metadata$description  
  if (!is.null(tdata))	datap$description[datap$ProjID == uniq_ids[idxproj]] <- tdata
  message(idxproj)
}
datap <- cbind(datap,t(rep(NA,768)))
for(idx in 1:nrow(datap))
{
	if(!is.na(datap$description[idx]))
	{
		datap[idx,9:776] <- embeddings(model = "nomic-embed-text:v1.5",prompt = datap$description[idx],host="http://192.168.11.98:9000")
	}
	message(idx)
}
# write_csv(datap,"~/Desktop/GitcoinCryptoPond/dataset/train.csv")

## Make IV and DV
datap_spl <- split(datap,datap$ProjID)
datapn <- do.call(rbind,lapply(datap_spl,function(x) cbind(data.frame(ProjID=x$ProjID[1],Amt=mean(x$Amt),MatchingPoolPct=sum(x$MatchingAmt)/sum(x$MatchingPool)),x[1,9:776])))
datapn <- datapn[!is.na(datapn$`1`),]
write_csv(datapn,"~/Desktop/GitcoinCryptoPond/dataset/train.csv")
