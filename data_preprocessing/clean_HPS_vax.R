rm(list = ls())

library(data.table)
library(lubridate)

setwd("~/Documents/Stanford/Projects/Multimodal Surveys/")

df <- NULL
for (i in 1:57) {
  message(i)
  if (i < 10) {
    temp <- fread(paste0("HPS/HPS_Week0", i, "_PUF_CSV/pulse2020_puf_0", i, ".csv"))
    temp <- temp[, AGE:=2020-TBIRTH_YEAR]
    temp <- temp[,.(SCRAM, WEEK, EST_ST, PWEIGHT, AGE, EGENDER, RHISPANIC, RRACE, EEDUC, MS, CURFOODSUF, FREEFOOD,
                    ANXIOUS, WORRY, INTEREST, DOWN, INCOME, THHLD_NUMPER, HLTHINS1, HLTHINS2, HLTHINS3, HLTHINS4,
                    HLTHINS5, HLTHINS6, HLTHINS7, HLTHINS8)]
    temp <- temp[, YEAR:=2020]
  } else if (i >= 10 & i <= 21) {
    temp <- fread(paste0("HPS/HPS_Week", i, "_PUF_CSV/pulse2020_puf_", i, ".csv"))  
    temp <- temp[, AGE:=2020-TBIRTH_YEAR]
    temp <- temp[,.(SCRAM, WEEK, EST_ST, PWEIGHT, AGE, EGENDER, RHISPANIC, RRACE, EEDUC, MS, CURFOODSUF, FREEFOOD,
                    ANXIOUS, WORRY, INTEREST, DOWN, INCOME, THHLD_NUMPER, HLTHINS1, HLTHINS2, HLTHINS3, HLTHINS4,
                    HLTHINS5, HLTHINS6, HLTHINS7, HLTHINS8)]
    temp <- temp[, YEAR:=2020]
  } else if (i >= 22 & i <=27) {
    temp <- fread(paste0("HPS/HPS_Week", i, "_PUF_CSV/pulse2021_puf_", i, ".csv"))  
    temp <- temp[, AGE:=2021-TBIRTH_YEAR]
    temp <- temp[,.(SCRAM, WEEK, EST_ST, PWEIGHT, AGE, EGENDER, RHISPANIC, RRACE, EEDUC, MS, CURFOODSUF, FREEFOOD,
                    ANXIOUS, WORRY, INTEREST, DOWN, INCOME, RECVDVACC, DOSES, GETVACC, THHLD_NUMPER, HLTHINS1, HLTHINS2, HLTHINS3, HLTHINS4,
                    HLTHINS5, HLTHINS6, HLTHINS7, HLTHINS8)]
    temp <- temp[, YEAR:=2021]
  } else if (i >= 28 & i <=33) {
      temp <- fread(paste0("HPS/HPS_Week", i, "_PUF_CSV/pulse2021_puf_", i, ".csv"))  
      temp <- temp[, AGE:=2021-TBIRTH_YEAR]
      temp <- temp[,.(SCRAM, WEEK, EST_ST, PWEIGHT, AGE, EGENDER, RHISPANIC, RRACE, EEDUC, MS, CURFOODSUF, FREEFOOD,
                      ANXIOUS, WORRY, INTEREST, DOWN, INCOME, RECVDVACC, DOSES, GETVACRV, THHLD_NUMPER, HLTHINS1, HLTHINS2, HLTHINS3, HLTHINS4,
                      HLTHINS5, HLTHINS6, HLTHINS7, HLTHINS8)]
      temp <- temp[, YEAR:=2021]
  } else if (i >= 34 & i <=39) {
    temp <- fread(paste0("HPS/HPS_Week", i, "_PUF_CSV/pulse2021_puf_", i, ".csv"))  
    temp <- temp[, AGE:=2021-TBIRTH_YEAR]
    temp <- temp[, EGENDER:=GENID_DESCRIBE]
    temp <- temp[EGENID_BIRTH%in%c(1,2) & EGENDER%in%c(-99, -88), EGENDER:=EGENID_BIRTH]
    temp <- temp[,.(SCRAM, WEEK, EST_ST, PWEIGHT, AGE, EGENDER, RHISPANIC, RRACE, EEDUC, MS, CURFOODSUF, FREEFOOD,
                    ANXIOUS, WORRY, INTEREST, DOWN, INCOME, RECVDVACC, DOSESRV, GETVACRV, THHLD_NUMKID, KIDS_LT5Y, KIDS_5_11Y, KIDS_12_17Y, KIDGETVAC, KIDDOSES, THHLD_NUMPER, HLTHINS1, HLTHINS2, HLTHINS3, HLTHINS4,
                    HLTHINS5, HLTHINS6, HLTHINS7, HLTHINS8)]
    temp <- temp[, YEAR:=2021]
  } else if (i == 40) {
    temp <- fread(paste0("HPS/HPS_Week", i, "_PUF_CSV/pulse2021_puf_", i, ".csv"))  
    temp <- temp[, AGE:=2021-TBIRTH_YEAR]
    temp <- temp[, EGENDER:=GENID_DESCRIBE]
    temp <- temp[EGENID_BIRTH%in%c(1,2) & EGENDER%in%c(-99, -88), EGENDER:=EGENID_BIRTH]
    temp <- temp[,.(SCRAM, WEEK, EST_ST, PWEIGHT, AGE, EGENDER, RHISPANIC, RRACE, EEDUC, MS, CURFOODSUF, FREEFOOD,
                    ANXIOUS, WORRY, INTEREST, DOWN, INCOME, RECVDVACC, NUMDOSES, BRAND, GETVACRV, THHLD_NUMKID, KIDS_LT5Y, KIDS_5_11Y, KIDS_12_17Y, KIDGETVAC, KIDDOSES,
                    NUMDOSES, BRAND, THHLD_NUMPER, HLTHINS1, HLTHINS2, HLTHINS3, HLTHINS4,
                    HLTHINS5, HLTHINS6, HLTHINS7, HLTHINS8)]
    temp <- temp[, YEAR:=2021]
  } else if (i >= 41 & i <=42) {
    temp <- fread(paste0("HPS/HPS_Week", i, "_PUF_CSV/pulse2022_puf_", i, ".csv"))  
    temp <- temp[, AGE:=2022-TBIRTH_YEAR]
    temp <- temp[, EGENDER:=GENID_DESCRIBE]
    temp <- temp[EGENID_BIRTH%in%c(1,2) & EGENDER%in%c(-99, -88), EGENDER:=EGENID_BIRTH]
    temp <- temp[,.(SCRAM, WEEK, EST_ST, PWEIGHT, AGE, EGENDER, RHISPANIC, RRACE, EEDUC, MS, CURFOODSUF, FREEFOOD,
                    ANXIOUS, WORRY, INTEREST, DOWN, INCOME, RECVDVACC, NUMDOSES, BRAND, GETVACRV, THHLD_NUMKID, KIDS_LT5Y, KIDS_5_11Y, KIDS_12_17Y, KIDGETVAC, KIDDOSES,
                    NUMDOSES, BRAND, THHLD_NUMPER, HLTHINS1, HLTHINS2, HLTHINS3, HLTHINS4,
                    HLTHINS5, HLTHINS6, HLTHINS7, HLTHINS8)]
    temp <- temp[, YEAR:=2022]
  } else if (i >= 43 & i <=45) {
    temp <- fread(paste0("HPS/HPS_Week", i, "_PUF_CSV/pulse2022_puf_", i, ".csv"))  
    temp <- temp[, AGE:=2022-TBIRTH_YEAR]
    temp <- temp[, EGENDER:=GENID_DESCRIBE]
    temp <- temp[EGENID_BIRTH%in%c(1,2) & EGENDER%in%c(-99, -88), EGENDER:=EGENID_BIRTH]
    temp <- temp[,.(SCRAM, WEEK, EST_ST, PWEIGHT, AGE, EGENDER, RHISPANIC, RRACE, EEDUC, MS, CURFOODSUF, FREEFOOD,
                    ANXIOUS, WORRY, INTEREST, DOWN, INCOME, RECVDVACC, NUMDOSES, BRAND, GETVACRV, THHLD_NUMKID, KIDS_LT5Y, KIDS_5_11Y, KIDS_12_17Y, KIDDOSES_5_11Y,
                    KIDDOSES_12_17Y, KIDGETVAC_5_11Y, KIDGETVAC_12_17Y,
                    NUMDOSES, BRAND, RBOOSTER, THHLD_NUMPER, HLTHINS1, HLTHINS2, HLTHINS3, HLTHINS4,
                    HLTHINS5, HLTHINS6, HLTHINS7, HLTHINS8)]
    temp <- temp[, YEAR:=2022]
  } else if (i >= 46 & i <=48) {
    temp <- fread(paste0("HPS/HPS_Week", i, "_PUF_CSV/pulse2022_puf_", i, ".csv"))  
    temp <- temp[, AGE:=2022-TBIRTH_YEAR]
    temp <- temp[, EGENDER:=GENID_DESCRIBE]
    temp <- temp[EGENID_BIRTH%in%c(1,2) & EGENDER%in%c(-99, -88), EGENDER:=EGENID_BIRTH]
    temp <- temp[,.(SCRAM, WEEK, EST_ST, PWEIGHT, AGE, EGENDER, RHISPANIC, RRACE, EEDUC, MS, CURFOODSUF, FREEFOOD,
                    ANXIOUS, WORRY, INTEREST, DOWN, INCOME, RECVDVACC, NUMDOSES, BOOSTERRV, THHLD_NUMKID, KIDS_LT5Y, KIDS_5_11Y, KIDS_12_17Y, 
                    KIDDOSESRV, KIDDOSESRV_5_11Y, KIDDOSESRV_12_17Y, KIDBSTR_5_11Y, KIDBSTR_12_17Y, KIDGETVAC_LT5Y, KIDGETVAC_5_11Y, KIDGETVAC_12_17Y, 
                    THHLD_NUMPER, HLTHINS1, HLTHINS2, HLTHINS3, HLTHINS4,
                    HLTHINS5, HLTHINS6, HLTHINS7, HLTHINS8)]
    temp <- temp[, YEAR:=2022]
  } else if (i >= 49 & i <=51) {
    temp <- fread(paste0("HPS/HPS_Week", i, "_PUF_CSV/pulse2022_puf_", i, ".csv"))  
    temp <- temp[, AGE:=2022-TBIRTH_YEAR]
    temp <- temp[, EGENDER:=GENID_DESCRIBE]
    temp <- temp[EGENID_BIRTH%in%c(1,2) & EGENDER%in%c(-99, -88), EGENDER:=EGENID_BIRTH]
    temp <- temp[,.(SCRAM, WEEK, EST_ST, PWEIGHT, AGE, EGENDER, RHISPANIC, RRACE, EEDUC, MS, CURFOODSUF, FREEFOOD,
                    ANXIOUS, WORRY, INTEREST, DOWN, INCOME, RECVDVACC, NUMDOSES, BOOSTERRV, THHLD_NUMKID, KIDS_LT5Y, KIDS_5_11Y, KIDS_12_17Y, 
                    KIDDOSESRV, KIDDOSESRV_LT5Y, KIDDOSESRV_5_11Y, KIDDOSESRV_12_17Y, KIDBSTR_5_11Y, KIDBSTR_12_17Y, KIDGETVAC_LT5Y, KIDGETVAC_5_11Y, KIDGETVAC_12_17Y, 
                    THHLD_NUMPER, HLTHINS1, HLTHINS2, HLTHINS3, HLTHINS4,
                    HLTHINS5, HLTHINS6, HLTHINS7, HLTHINS8)]
    temp <- temp[, YEAR:=2022]
  } else if (i == 52) {
    temp <- fread(paste0("HPS/HPS_Week", i, "_PUF_CSV/pulse2022_puf_", i, ".csv")) 
    temp <- temp[, AGE:=2022-TBIRTH_YEAR]
    temp <- temp[, EGENDER:=GENID_DESCRIBE]
    temp <- temp[EGENID_BIRTH%in%c(1,2) & EGENDER%in%c(-99, -88), EGENDER:=EGENID_BIRTH]
    temp <- temp[,.(SCRAM, WEEK, EST_ST, PWEIGHT, AGE, EGENDER, RHISPANIC, RRACE, EEDUC, MS, CURFOODSUF, FREEFOOD,
                    ANXIOUS, WORRY, INTEREST, DOWN, INCOME, RECVDVACC, THHLD_NUMPER, HLTHINS1, HLTHINS2, HLTHINS3, HLTHINS4,
                    THHLD_NUMKID, KIDS_LT5Y, KIDS_5_11Y, KIDS_12_17Y, KIDVACWHEN_LT5Y, KIDVACWHEN_5_11Y, KIDVACWHEN_12_17Y, 
                    KIDGETVAC_LT5Y, KIDGETVAC_5_11Y,KIDGETVAC_12_17Y,
                    HLTHINS5, HLTHINS6, HLTHINS7, HLTHINS8)]
    temp <- temp[, YEAR:=2022]
  } else {
    temp <- fread(paste0("HPS/HPS_Week", i, "_PUF_CSV/pulse2023_puf_", i, ".csv")) 
    temp <- temp[, AGE:=2023-TBIRTH_YEAR]
    temp <- temp[, EGENDER:=GENID_DESCRIBE]
    temp <- temp[EGENID_BIRTH%in%c(1,2) & EGENDER%in%c(-99, -88), EGENDER:=EGENID_BIRTH]
    temp <- temp[,.(SCRAM, WEEK, EST_ST, PWEIGHT, AGE, EGENDER, RHISPANIC, RRACE, EEDUC, MS, CURFOODSUF, FREEFOOD,
                    ANXIOUS, WORRY, INTEREST, DOWN, INCOME, RECVDVACC, THHLD_NUMPER, HLTHINS1, HLTHINS2, HLTHINS3, HLTHINS4,
                    THHLD_NUMKID, KIDS_LT5Y, KIDS_5_11Y, KIDS_12_17Y, KIDVACWHEN_LT5Y, KIDVACWHEN_5_11Y, KIDVACWHEN_12_17Y, KIDGETVAC_LT5Y,
                    KIDGETVAC_5_11Y,KIDGETVAC_12_17Y,
                    HLTHINS5, HLTHINS6, HLTHINS7, HLTHINS8)]
    temp <- temp[, YEAR:=2023]
  }
  df <- rbind(df, temp, fill = T)
}

## 34-39: 12-17 only
## 40-42: mixed, no distinction
## 43-57: specific

df <- df[KIDS_LT5Y==1, kid_lt5_denom:=1]
df <- df[KIDS_LT5Y== -99 | THHLD_NUMKID==0, kid_lt5_denom:=0]
df <- df[KIDS_5_11Y==1, kid_5_11_denom:=1]
df <- df[KIDS_5_11Y== -99 | THHLD_NUMKID==0, kid_5_11_denom:=0]
df <- df[KIDS_12_17Y==1, kid_12_17_denom:=1]
df <- df[KIDS_12_17Y== -99 | THHLD_NUMKID==0, kid_12_17_denom:=0]

## 12-17
df <- df[WEEK>=34 & WEEK <= 42 & (KIDDOSES==1 | KIDGETVAC == 1 | KIDGETVAC == 2) & kid_12_17_denom==1, vax_probdef_12_17 := 1]
df <- df[WEEK>=34 & WEEK <= 42 & (KIDGETVAC %in% c(3, 4, 5)) & kid_12_17_denom==1, vax_probdef_12_17 := 0]
df <- df[WEEK>=43 & WEEK <= 45 & (KIDDOSES_12_17Y==1 | KIDGETVAC_12_17Y==1 | KIDGETVAC_12_17Y==2) & kid_12_17_denom==1, vax_probdef_12_17:=1]
df <- df[WEEK>=43 & WEEK <= 45 & (KIDGETVAC_12_17Y %in% c(3, 4, 5)) & kid_12_17_denom==1, vax_probdef_12_17:=0]
df <- df[WEEK>=46 & WEEK <= 51 & ((KIDDOSESRV == 1 & KIDDOSESRV_12_17Y==1) | KIDGETVAC_12_17Y==1 | KIDGETVAC_12_17Y==2) & kid_12_17_denom==1, vax_probdef_12_17:=1]
df <- df[WEEK>=46 & WEEK <= 51 & (KIDGETVAC_12_17Y %in% c(3, 4, 5) | KIDDOSESRV_12_17Y==2) & kid_12_17_denom==1, vax_probdef_12_17:=0]
df <- df[WEEK>=52 & WEEK <= 57 & (KIDVACWHEN_12_17Y %in% c(1, 2, 3) | KIDGETVAC_12_17Y==1 | KIDGETVAC_12_17Y==2) & kid_12_17_denom==1, vax_probdef_12_17:=1]
df <- df[WEEK>=52 & WEEK <= 57 & (KIDGETVAC_12_17Y %in% c(3, 4, 5)) & kid_12_17_denom==1, vax_probdef_12_17:=0]

## 5-11
df <- df[WEEK>=40 & WEEK <= 42 & (KIDDOSES==1 | KIDGETVAC == 1 | KIDGETVAC == 2) & kid_5_11_denom==1, vax_probdef_5_11 := 1]
df <- df[WEEK>=40 & WEEK <= 42 & (KIDGETVAC %in% c(3, 4, 5)) & kid_5_11_denom==1, vax_probdef_5_11 := 0]
df <- df[WEEK>=43 & WEEK <= 45 & (KIDDOSES_5_11Y==1 | KIDGETVAC_5_11Y==1 | KIDGETVAC_5_11Y==2) & kid_5_11_denom==1, vax_probdef_5_11:=1]
df <- df[WEEK>=43 & WEEK <= 45 & (KIDGETVAC_5_11Y %in% c(3, 4, 5)) & kid_5_11_denom==1, vax_probdef_5_11:=0]
df <- df[WEEK>=46 & WEEK <= 51 & ((KIDDOSESRV == 1 & KIDDOSESRV_5_11Y==1) | KIDGETVAC_5_11Y==1 | KIDGETVAC_5_11Y==2) & kid_5_11_denom==1, vax_probdef_5_11:=1]
df <- df[WEEK>=46 & WEEK <= 51 & (KIDGETVAC_5_11Y %in% c(3, 4, 5) | KIDDOSESRV_5_11Y==2) & kid_5_11_denom==1, vax_probdef_5_11:=0]
df <- df[WEEK>=52 & WEEK <= 57 & (KIDVACWHEN_5_11Y %in% c(1, 2, 3) | KIDGETVAC_5_11Y==1 | KIDGETVAC_5_11Y==2) & kid_5_11_denom==1, vax_probdef_5_11:=1]
df <- df[WEEK>=52 & WEEK <= 57 & (KIDGETVAC_5_11Y %in% c(3, 4, 5)) & kid_5_11_denom==1, vax_probdef_5_11:=0]

## LT5
df <- df[WEEK>=49 & WEEK <= 51 & ((KIDDOSESRV == 1 & KIDDOSESRV_LT5Y==1) | KIDGETVAC_LT5Y==1 | KIDGETVAC_LT5Y==2) & kid_lt5_denom==1, vax_probdef_lt5:=1]
df <- df[WEEK>=49 & WEEK <= 51 & (KIDGETVAC_LT5Y %in% c(3, 4, 5) | KIDDOSESRV_LT5Y==2) & kid_lt5_denom==1, vax_probdef_lt5:=0]
df <- df[WEEK>=52 & WEEK <= 57 & (KIDVACWHEN_LT5Y %in% c(1, 2, 3) | KIDGETVAC_LT5Y==1 | KIDGETVAC_LT5Y==2) & kid_lt5_denom==1, vax_probdef_lt5:=1]
df <- df[WEEK>=52 & WEEK <= 57 & (KIDGETVAC_LT5Y %in% c(3, 4, 5)) & kid_lt5_denom==1, vax_probdef_lt5:=0]

df <- df[AGE <= 24 & AGE >=18, age_grp:=1]
df <- df[AGE >= 25 & AGE <= 29, age_grp:=2]
df <- df[AGE >= 30 & AGE <= 34, age_grp:=3]
df <- df[AGE >= 35 & AGE <= 39, age_grp:=4]
df <- df[AGE >= 40 & AGE <= 44, age_grp:=5]
df <- df[AGE >= 45 & AGE <= 49, age_grp:=6]
df <- df[AGE >= 50 & AGE <= 54, age_grp:=7]
df <- df[AGE >= 55 & AGE <= 59, age_grp:=8]
df <- df[AGE >= 60 & AGE <= 64, age_grp:=9]
df <- df[AGE >= 65 & AGE <= 69, age_grp:=10]
df <- df[AGE >= 70 & AGE <= 74, age_grp:=11]
df <- df[AGE >= 75 & AGE <= 79, age_grp:=12]
df <- df[AGE >= 80, age_grp:=13]

df <- df[EGENDER==1, male:=1]
df <- df[EGENDER==2, male:=0]

df <- df[RHISPANIC==2, race_eth:="Hispanic"]
df <- df[is.na(race_eth) & RRACE==1, race_eth:="NH White"]
df <- df[is.na(race_eth) & RRACE==2, race_eth:="NH Black"]
df <- df[is.na(race_eth) & RRACE==3, race_eth:="NH Asian"]
df <- df[is.na(race_eth) & RRACE==4, race_eth:="NH Other"]

df <- df[INCOME==1, income_brfss:=1]
df <- df[INCOME==2, income_brfss:=2]
df <- df[INCOME==3, income_brfss:=3]
df <- df[INCOME==4, income_brfss:=4]
df <- df[INCOME%in%c(5, 6, 7, 8), income_brfss:=5]

df <- df[INCOME <= 8 & !(INCOME%in%c(-99, -88)), income_detailed:=INCOME]

df <- df[EEDUC%in%c(1, 2), edu_cat:=1]
df <- df[EEDUC==3, edu_cat:=2]
df <- df[EEDUC%in%c(4, 5), edu_cat:=3]
df <- df[EEDUC%in%c(6, 7), edu_cat:=4]

df <- df[ANXIOUS %in% c(1,2,3,4) & WORRY %in% c(1,2,3,4) & INTEREST %in% c(1,2,3,4) & DOWN %in% c(1,2,3,4) , sum_mh:=ANXIOUS+WORRY+INTEREST+DOWN]
df <- df[ANXIOUS %in% c(-99, -88), ANXIOUS:=NA]
df <- df[WORRY %in% c(-99, -88), WORRY:=NA]
df <- df[INTEREST %in% c(-99, -88), INTEREST:=NA]
df <- df[DOWN %in% c(-99, -88), DOWN:=NA]

df <- df[RECVDVACC %in% c(-99, -88), RECVDVACC:=NA]
df <- df[RECVDVACC==2, RECVDVACC:=0]

df <- df[, one_or_more_vax:=RECVDVACC]
df <- df[one_or_more_vax==1 & DOSES==1, complete:=1]
df <- df[one_or_more_vax==1 & DOSES==2, complete:=0]

df <- df[RECVDVACC==1 & !is.na(GETVACRV), vax_attitude5cat_probdef:=1]
df <- df[GETVACRV%in%c(1,2), vax_attitude5cat_probdef:=1]
df <- df[GETVACRV%in%c(3,4,5), vax_attitude5cat_probdef:=0]

df <- df[RECVDVACC==1 & !is.na(GETVACRV), vax_attitude5cat_def:=1]
df <- df[GETVACRV%in%c(1), vax_attitude5cat_def:=1]
df <- df[GETVACRV%in%c(2,3,4,5), vax_attitude5cat_def:=0]

df <- df[RECVDVACC==1 & !is.na(GETVACC), vax_attitude4cat_probdef:=1]
df <- df[GETVACC%in%c(1,2), vax_attitude4cat_probdef:=1]
df <- df[GETVACC%in%c(3,4), vax_attitude4cat_probdef:=0]

df <- df[RECVDVACC==1 & !is.na(GETVACC), vax_attitude4cat_def:=1]
df <- df[GETVACC%in%c(1), vax_attitude4cat_def:=1]
df <- df[GETVACC%in%c(2,3,4), vax_attitude4cat_def:=0]

df <- df[RECVDVACC==1, vax_cat:="Vaccinated"]
df <- df[GETVACC==1, vax_cat:="Definitely Yes"]
df <- df[GETVACC==2, vax_cat:="Probably Yes"]
df <- df[GETVACC==3, vax_cat:="Probably No"]
df <- df[GETVACC==4, vax_cat:="Definitely No"]
df <- df[GETVACRV==1, vax_cat:="Definitely Yes"]
df <- df[GETVACRV==2, vax_cat:="Probably Yes"]
df <- df[GETVACRV==3, vax_cat:="Unsure"]
df <- df[GETVACRV==4, vax_cat:="Probably No"]
df <- df[GETVACRV==5, vax_cat:="Definitely No"]

df <- df[MS==1, marital:="Married"]
df <- df[MS==2, marital:="Widowed"]
df <- df[MS%in% c(3, 4), marital:="Divorced/Separated"]
df <- df[MS==5, marital:="Never Married"]

setnames(df, "THHLD_NUMPER", "household_size")

df <- df[HLTHINS1==1, emp_ins:=1]
df <- df[(HLTHINS1==2 | HLTHINS1==-99), emp_ins:=0]
df <- df[HLTHINS3==1, medicare_ins:=1]
df <- df[(HLTHINS3==2 | HLTHINS3==-99), medicare_ins:=0]
df <- df[HLTHINS4==1, medicaid_ins:= 1]
df <- df[(HLTHINS4==2 | HLTHINS4==-99), medicaid_ins:= 0]
df <- df[HLTHINS2==1 | HLTHINS5==1 | HLTHINS6==1 | HLTHINS7==1 | HLTHINS8==1, other_ins:=1]
df <- df[(HLTHINS2==2 | HLTHINS2==-99) & (HLTHINS5==2 | HLTHINS5==-99) & 
           (HLTHINS6==2 | HLTHINS6==-99) & (HLTHINS7==2 | HLTHINS7==-99) & (HLTHINS8==2 | HLTHINS8==-99), other_ins:=0]

df <- df[(emp_ins==1 & !is.na(emp_ins)) | (medicare_ins==1 & !is.na(medicare_ins)) |
           (medicaid_ins==1 & !is.na(medicaid_ins)) | (other_ins==1 & !is.na(other_ins)), any_ins:=1]
df <- df[emp_ins==0 & medicare_ins == 0 & medicaid_ins == 0 & other_ins == 0, any_ins:=0]

fips <- as.data.table(tidycensus::fips_codes)
fips <- unique(fips[,.(state, state_code, state_name)])
fips <- fips[, state_code:=as.numeric(state_code)]

df <- merge(df, fips, by.x = "EST_ST", by.y = "state_code", all.x=T)

df <- df[,.(YEAR, SCRAM, WEEK, PWEIGHT, ANXIOUS, WORRY, INTEREST, DOWN, sum_mh, RECVDVACC, age_grp, 
            male, race_eth, income_brfss, income_detailed, edu_cat, state_name, marital, 
            household_size, emp_ins, medicare_ins, medicaid_ins, other_ins, any_ins,
            vax_attitude5cat_probdef, vax_attitude5cat_def, vax_attitude4cat_probdef, vax_attitude4cat_def, vax_cat,
            vax_probdef_12_17, vax_probdef_5_11, vax_probdef_lt5)]

dates <- as.data.table(readxl::read_xlsx("hps_weeks_lookup.xlsx"))
setnames(dates, "Week", "WEEK")
dates <- dates[, Start:=ymd(Start)]
dates <- dates[, End:=ymd(End)]

expanded_dates <- dates[, .(date = seq(Start, End, by = "days")), by = "WEEK"]
setnames(expanded_dates, "date", "Date")

df <- merge(df, dates, by = c("WEEK"))
df <- df[, mid_date:=Start + (End - Start)/2]

week_mid_date <- unique(df[,.(WEEK, mid_date)])

df <- df[vax_cat == "Vaccinated", vax_cat:=1]
df <- df[vax_cat == "Definitely Yes", vax_cat:=2]
df <- df[vax_cat == "Probably Yes", vax_cat:=3]
df <- df[vax_cat == "Unsure", vax_cat:=4]
df <- df[vax_cat == "Probably No", vax_cat:=5]
df <- df[vax_cat == "Definitely No", vax_cat:=6]
df <- df[, vax_cat:=as.numeric(vax_cat)]

df <- df[, vax_cat:=NULL]
df <- df[!is.na(vax_attitude4cat_def), vax_probdef_adult:=vax_attitude4cat_probdef]
df <- df[!is.na(vax_attitude5cat_def), vax_probdef_adult:=vax_attitude5cat_probdef]

write.csv(df, "HPS/hps_prepped_date_novaxcat.csv", na = "", row.names = F)
