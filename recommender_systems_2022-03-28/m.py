import pandas as pd
import visidata as vd

fn = "MEMBERS_ACTIVITY.csv"
#df = pd.read_csv(fn, nrows=1)
df = pd.read_csv("xxx1.csv", nrows=100, encoding = "ISO-8859-1")
#df = pd.read_csv("xxx1.csv", nrows=100000, encoding = "ISO-8859-1")
print(len(df.columns))
print(df.columns[37], df.columns[46])
cols = df.columns
col37 = df[cols[37]]
col46 = df[cols[46]]
print(col37)
print(col46)
#df.to_csv("activity_top10e5.csv", index=0, encoding='utf-8')
