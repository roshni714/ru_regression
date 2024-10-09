library(tidycensus)
library(magrittr)
library(data.table)

setwd("~/Documents/Stanford/Projects/Multimodal Surveys/")

vars <- load_variables(year = 2022, dataset = "acs5/profile", cache = TRUE)

var_list <- c("DP02_0016", "DP02_0026P", "DP02_0032P", "DP02_0040", "DP02_0060P", "DP02_0061P", "DP02_0062P", "DP02_0063P", "DP02_0064P",
              "DP02_0065P", "DP02_0066P", "DP02_0070P", "DP02_0089P", "DP03_0009P",
              "DP02_0113P", "DP02_0153P", "DP02_0154P", "DP03_0062", "DP03_0063", "DP03_0074P",
              "DP03_0096P", "DP03_0097P","DP03_0128P", "DP04_0089", "DP04_0134")

acs <- get_acs(geography = "state", variables = var_list, cache_table = TRUE, year = 2022)

acs <- merge(acs, vars, by.x = "variable", by.y = "name")

acs <- as.data.table(acs)

for (i in unique(acs$variable)) {
  message(i, ": ", unique(acs$label[acs$variable==i]))
}

acs <- acs[variable == "DP02_0016", var:="avg_hh_size"]
acs <- acs[variable == "DP02_0026P", var:="male_never_married"]
acs <- acs[variable == "DP02_0032P", var:="female_never_married"]
acs <- acs[variable == "DP02_0040", var:="fertility_rate"]
acs <- acs[variable %in% c("DP02_0060P", "DP02_0061P", "DP02_0062P"), var:="edu_hs_or_less"]
acs <- acs[variable %in% c("DP02_0063P", "DP02_0064P"), var:="some_college_or_2yr"]
acs <- acs[variable %in% c("DP02_0065P"), var:="4yr_college"]
acs <- acs[variable %in% c("DP02_0066P"), var:="graduate_degree"]
acs <- acs[variable == "DP02_0070P", var:="veterans"]
acs <- acs[variable == "DP02_0089P", var:="us_born"]
acs <- acs[variable == "DP02_0113P", var:="english_only"]
acs <- acs[variable == "DP02_0153P", var:="hh_computer"]
acs <- acs[variable == "DP02_0154P", var:="hh_internet"]
acs <- acs[variable == "DP03_0009P", var:="unemployment_rate"]
acs <- acs[variable == "DP03_0062", var:="median_income"]
acs <- acs[variable == "DP03_0063", var:="mean_income"]
acs <- acs[variable == "DP03_0074P", var:="food_stamps"]
acs <- acs[variable == "DP03_0096P", var:="any_health_insurance"]
acs <- acs[variable == "DP03_0097P", var:="private_health_insurance"]
acs <- acs[variable == "DP03_0128P", var:="poverty"]
acs <- acs[variable == "DP04_0089", var:="median_house_value"]
acs <- acs[variable == "DP04_0134", var:="median_rent"]

acs <- acs[, .(var, GEOID, NAME, estimate)]
acs <- acs[, lapply(.SD, sum), by = c("var", "GEOID", "NAME")]

acs <- dcast(acs, GEOID + NAME ~ var, value.var = "estimate")
acs <- acs[!(NAME%like%"Puerto Rico")]

fips <- as.data.table(tidycensus::fips_codes)
fips <- unique(fips[,.(state_code, state, state_name)])
fips <- fips[, GEOID:=paste0(state_code)]
acs <- merge(acs, fips, by = "GEOID")

state_voting <- fread("2020_president.csv")
acs <- merge(acs, state_voting[,.(state_po, vote_pct)], by.x = "state", by.y = "state_po")
setnames(acs, "vote_pct", "republican_pct")
acs <- acs[, republican_pct:=republican_pct*100]

acs <- acs[,c("GEOID", "NAME"):=NULL]

total_population <- as.data.table(get_decennial(
  geography = "state", 
  variables = "P1_001N",
  year = 2020)) %>%
  setnames(., "value", "tot_pop")

acs <- merge(acs, total_population[,.(GEOID, tot_pop)], by.x="state_code", by.y = "GEOID", all.x=T)

write.csv(acs, "acs_state_characteristics.csv", na = "", row.names = F)



