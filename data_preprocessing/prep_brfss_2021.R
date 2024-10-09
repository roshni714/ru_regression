rm(list = ls())

library(tidyverse)
library(data.table)

setwd("~/Documents/Stanford/Projects/Multimodal Surveys")

df <- haven::read_xpt("LLCP2021.XPT")
df <- as.data.table(df)

## KEEP A SUBSET OF RELEVANT VARIABLES
df <- df[,.(`_STATE`, SEQNO, `_PSU`, `_STSTR`, `_LLCPWT`, `_LTASTH1`, DIABETE4, `_AGEG5YR`, `_SEX`, `_RACE`, `_BMI5`, `INCOME3`, `EDUCA`, MENTHLTH, `_SMOKER3`,
            HHADULT,  CHILDREN, MARITAL, `_HLTHPLN`, PRIMINSR)]

colnames(df) <- gsub(x = colnames(df), pattern = "_", replacement = "")

fips <- as.data.table(tidycensus::fips_codes)
fips <- unique(fips[,.(state_name, state_code)])
fips <- fips[, state_code:=as.numeric(state_code)]

df <- merge(df, fips, by.x = "STATE", by.y = "state_code")

df <- df[LTASTH1==1, asthma:=0]
df <- df[LTASTH1==2, asthma:=1]

setnames(df, "AGEG5YR", "age_grp")
df <- df[age_grp==14, age_grp:=NA]

df <- df[SEX==1, male:=1]
df <- df[SEX==2, male:=0]

df <- df[RACE==1, race_eth:="NH White"]
df <- df[RACE==2, race_eth:="NH Black"]
df <- df[RACE==3, race_eth:="NH AIAN"]
df <- df[RACE==4, race_eth:="NH Asian"]
df <- df[RACE==5, race_eth:="NH NHPI"]
df <- df[RACE==6, race_eth:="NH Other"]
df <- df[RACE==7, race_eth:="NH Multiple"]
df <- df[RACE==8, race_eth:="Hispanic"]

df <- df[, bmi:=BMI5/100]

df <- df[INCOME3 %in% c(1, 2, 3, 4), income_brfss:=1]
df <- df[INCOME3%in%c(5), income_brfss:=2]
df <- df[INCOME3%in%c(6), income_brfss:=3]
df <- df[INCOME3%in%c(7), income_brfss:=4]
df <- df[INCOME3%in%c(8, 9, 10, 11), income_brfss:=5]

df <- df[INCOME3 %in% c(1, 2, 3, 4), income_detailed:=1]
df <- df[INCOME3%in%c(5), income_detailed:=2]
df <- df[INCOME3%in%c(6), income_detailed:=3]
df <- df[INCOME3%in%c(7), income_detailed:=4]
df <- df[INCOME3%in%c(8), income_detailed:=5]
df <- df[INCOME3%in%c(9), income_detailed:=6]
df <- df[INCOME3%in%c(10), income_detailed:=7]
df <- df[INCOME3%in%c(11), income_detailed:=8]

setnames(df, "MENTHLTH", "mental_health")

df <- df[mental_health == 88, mental_health:=0]
df <- df[mental_health > 30, mental_health:=NA]

df <- df[SMOKER3%in%c(1,2), smoker:=1]
df <- df[SMOKER3%in%c(3,4), smoker:=0]

df <- df[, edu_cat:=EDUCA]
df <- df[edu_cat%in%c(1,2, 3), edu_cat:= 1]
df <- df[edu_cat%in%c(4), edu_cat:= 2]
df <- df[edu_cat%in%c(5), edu_cat:= 3]
df <- df[edu_cat%in%c(6), edu_cat:= 4]
df <- df[edu_cat==9, edu_cat:=NA]

df <- df[DIABETE4==1, diabetes:=1]
df <- df[DIABETE4%in%c(2, 3, 4), diabetes:=0]

df <- df[MARITAL==1, marital:="Married"]
df <- df[MARITAL==3, marital:="Widowed"]
df <- df[MARITAL%in% c(2, 4), marital:="Divorced/Separated"]
df <- df[MARITAL%in%c(5, 6), marital:="Never Married"]

df <- df[HHADULT%in% c(77, 99), HHADULT:=NA]
df <- df[CHILDREN%in% c(88), CHILDREN:=0]
df <- df[CHILDREN%in% c(99), CHILDREN:=NA]

df <- df[, household_size:=HHADULT+CHILDREN]
df <- df[household_size>10, household_size:=10]

df <- df[HLTHPLN==1, any_ins:=1]
df <- df[HLTHPLN==2, any_ins:=0]

df <- df[!(PRIMINSR%in%c(77, 99)), emp_ins:=ifelse(PRIMINSR==1, 1, 0)]
df <- df[!(PRIMINSR%in%c(77, 99)), medicare_ins:=ifelse(PRIMINSR==3, 1, 0)]
df <- df[!(PRIMINSR%in%c(77, 99)), medicaid_ins:=ifelse(PRIMINSR==5, 1, 0)]
df <- df[!(PRIMINSR%in%c(77, 99)), other_ins:=ifelse(PRIMINSR%in%c(4, 6, 7, 8, 9, 10), 1, 0)]

df <- df[, .(SEQNO, PSU, STSTR, LLCPWT, state_name, age_grp, male, edu_cat, income_brfss, income_detailed, race_eth, smoker, diabetes, 
             asthma, bmi, mental_health, any_ins, emp_ins, medicare_ins, medicaid_ins, other_ins, marital, household_size)]

write.csv(df, "brfss_2021_num.csv", na = "", row.names = F)

