import pandas as pd
import date_library as datelib

def clean_dataframe(df0):
    df = df0.copy()
    df.fillna(-1, inplace=True)
    float_cols = df.columns[df.dtypes == 'float']
    df[float_cols] = df[float_cols].astype('int')   # recommended use of view generates error
    # It would nice to identify the dates automatically
    date_cols = ['ENROLL_DATE','LAST_TIER_CHANGE_DATE', 'BIRTH_DATE', 'ACTIVITY_DATE', 'FLIGHT_DATE', 'BOOKING_DATE', 'TICKET_SALES_DATE']

    for col in date_cols: 
        df[col] = datelib.date_to_timestamp(df[col])

    df.iloc[0].T.head(50)
    return df
#-----------------------------------------------------------
