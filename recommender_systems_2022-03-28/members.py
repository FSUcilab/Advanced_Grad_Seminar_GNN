import pandas as pd
import visidata as vd

df = pd.read_csv("MEMBERS_CATALOG.csv")

print(df.columns)
dfg_id = df.groupby("MEMBER_ID").size()

# There is a single line per member
print(dfg_id.max())

# MEMBER_ID,
# TIER_LEVEL,
# TIER_LEVEL_DESCRIPTION,
# PREVIOUS_TIER,
# LAST_TIER_CHANGE_DATE,  xx
# STATUS,
# ENROLL_DATE,  xx
# GENDER,
# BIRTH_DATE,   xx
# NATIONALITY,
# ADDR_COUNTRY,
# COMMUNICATION_LANGUAGE,
# CONNECTMILES_OFFERS_AND_NEWS,
# E_STATEMENTS,
# COPA_AIRLINES_OFFERS_AND_NEWS,
# CCD_ACTIVE_IND,
# MILES_BALANCE,
# TOTAL_MILES_SINCE_ENROLLMENT

print(df.describe().T)

# Change dates to nb seconds since 1970
# LAST_TIER_CHANGE_DATE
# ENROLL_DATE
# BIRTH_DATE

# Categorical distributions for: 
#   Nationality
#   ADDR_COUNTRY
#   COMMUNICATION LANG
#   GENDER
#   STATUS
#   PREVIOUS_TIER

# What is T1 versus T1CC which are both Silver. WHY? 

for col in df.columns: 
    print(df[col].unique())

for col in df.columns: 
    print(f"{col}: ", df[col].nunique())

fn = "MEMBERS_ACTIVITY.csv"
#df = pd.read_csv(fn, nrows=1)
df = pd.read_csv("xxx1.csv", nrows=100000) # encoding='utf-8') # nrows=2)
#df.to_csv("activity_top10e5.csv", index=0)
df.to_csv("yyy.csv", index=0)
