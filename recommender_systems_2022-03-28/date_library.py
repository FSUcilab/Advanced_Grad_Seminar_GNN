#!/usr/bin/env python
# coding: utf-8

# Developed for Copa Airlines

import pandas as pd
from datetime import date, datetime
from dateutil import parser
from dateutil.tz import gettz
import pytz
import time 
import regex as rex

from IPython.display import display

#------------------------------------------------------
def date_to_timestamp(date_series):
    # input date strings (PTY time)
    # return: date timestamps in seconds
    # date after conversion via pd.to_datetime
    return pd.to_datetime(date_series).astype('int') // 1000000000
#------------------------------------------------------
def timestampToDateTimeUTC(ts):
    # ts: timestamp in seconds
    #dt_tm = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
    dt_tm = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts))
    return dt_tm[0:10], dt_tm[11:16]
#---------------------------------------------------------
def timestampToDateTimePTY(ts):
    # ts: timestamp in seconds
    dt_tm = timestampToDateTimeUTC(ts + 5*3600)
    return dt_tm
#---------------------------------------------------------
def dateTimeUTCToTimestamp(date='2019/10/01', tim='20:00'): #, UTC_offset='0'):
    timestamp = 0;
    date = date + tim + "+0000"

    # convert string to datetimeformat
    date = datetime.strptime(date, "%Y/%m/%d%H:%M%z")

    # convert datetime to timestamp
    return int(datetime.timestamp(date))
#---------------------------------------------------------
def dateTimePTYToTimestamp(date='2019/10/01', tim='20:00'): #, UTC_offset='0'):
    timestamp = 0;
    date = date + tim + "+0500"  # PTY is 5 hrs ahead of UTC

    # convert string to datetimeformat
    #date = datetime.strptime(date, "%Y/%m/%d%H:%M%z")  # in CRA
    date = datetime.strptime(date, "%Y-%m-%d%H:%M%z") # in COPA

    # convert datetime to timestamp
    return int(datetime.timestamp(date))
#------------------------------------------------------------
def to_datetime(series):
    series_tmz = series.str[-12:-7]
    series_dtz = pd.to_datetime(series)
    return pd.concat([series_dtz, series_tmz], axis=1)
#------------------------------------------------------
def Zulu2PTY(datetime):
    """
    Convert from Zulu to PTY date-time.

    Parameters: 
        datetime [Series]: a series of date/times stored in nanoseconds integer format

    Return: 
        A series of date/times in nanoseconds integer format in Panama city

    If it is 1:00 pm Zulu, it is 6:00 pm PTY time (DTML)
    """

    # PTY time is earlier than Zulu time by 5 hours. However, data from the web suggests 
    # that PTY is 5 hours later than Zulu, and even 4 hours later according to one site. I have no idea what the truth is. 
    # According to Miguel, Panama is always 5 hours BEHIND Zulu

    return datetime - 5 * 3600 * 1000000000 # 5 hours difference # clutching at straws
#--------------------------------------------------
# Standard time :  UTC-5.0 
# Daylight time :  UTC-4.0

"""
tzinfo = {
          "EDT": pytz.timezone('US/Eastern'),
          "CDT": pytz.timezone('US/Central'),
          "MDT": pytz.timezone('US/Mountain'),
          "PDT": pytz.timezone('US/Pacific'), 
          "HDT": pytz.timezone('US/Hawaii'), 
          "GMT": pytz.timezone('UTC'),
    
          # We will convert all times to UTC, then to seconds since 1970
          # Then we will add/subtrace one hour (3600 sec) if time is in 
          # Savings Time (as opposed to Daylight Time)
          "EST": pytz.timezone('US/Eastern'),
          "CST": pytz.timezone('US/Central'),
          "MST": pytz.timezone('US/Mountain'),
          "PST": pytz.timezone('US/Pacific'), 
          "HST": pytz.timezone('US/Hawaii'),  
         }
 """

# For example: when Eastern Daylight Time is used, the international meetings are held 
# between 10 am to 6 pm whereas for Eastern Standard Time the same meeting and conference 
# are held between 9 am to 5 pm.  So, 10 am EDT  <=====> 9 am EST

#--------------------------------------------------------------------------------------
def normalize(name1, exceptions, date_dict, time_zone, tzinfo):
    # try: 
    # flag = False
    
    try:
        if type(name1) != str:
            date_dict.append((name1, '', ''))
            time_zone.append('')
            return

        if rex.match(r'.*GMT', name1):   # missing some matches. WHY?
            # print(type(name), name)//
            if rex.match(r'.*2\/16\/17 ', name1): flag = True
            name = rex.sub(r'\(GMT-05:00\)', ' EST ', name1)
            name = rex.sub(r'\(GMT-04:00\)', ' EDT ', name)
            name = rex.sub(r'(\d+?\/\d+?)\/(\d\d )', r'\1/20\2', name)  # will fail in the year 21xx <<<< NOT WORKING
        else:
            name = name1

        # Missing 4-digit year
        if not rex.match(r'.*\d{4}', name):
            date_dict.append((name1, '', ''))
            time_zone.append('')
            return

        name = rex.sub(r'(Febnlaiy|Febnlaly|Febiuaiy|Feb1ua1y|Februa ry)', 'February', name)
        name = rex.sub(r'(Janualy|J anuary)', 'January', name)
        name = rex.sub(r"(\'iuesday)", 'Tuesday', name) 
        name = rex.sub(r'Septem ber', 'September', name)

        name = rex.sub(r',(\d{4})', r', \1', name)
        name = rex.sub(r'(\d{4})(\d\d:)', r'\1 \2', name)
        name = rex.sub(r'(\d{4} \d) :', r'\1:', name) 
        name = rex.sub(r'(\d?:\d?:) (\d?)', r'\1\2', name)
        name = rex.sub(r'(\d{4} \d+:) (\d)', r'\1\2', name)  # CHECK ORIGINAL TEXT. Why does this error occur?
        # 12:42: 22 ==> 12:42:22
        name = rex.sub(r'(\d\d:\d\d:)\ (\d\d)', r'\1\2', name)  # CHECK ORIGINAL TEXT. Why does this error occur?
        name = rex.sub(r'(\w+ \d+):( \d{4})', r'\1, \2', name)
        name = rex.sub(r'\>', ':', name)
        name = rex.sub(r'\?', '', name)

        tz = 'EST'
        if rex.match('.*Eastern Daylight Time', name):
            name = rex.sub('Eastern Daylight Time', 'EDT', name)  # difference between Daylight and Standard
            tz = 'EDT'
        elif rex.match('.*Eastern Standard Time', name):
            name = rex.sub('Eastern Standard Time', 'EST', name)
            tz = 'EST'
        elif rex.match('.*PST', name):
            tz = 'PST'
        elif rex.match('.*PDT', name):   # Pacific
            tz = 'PDT'
        elif rex.match('.*HST', name):   # Hawaii
            tz = 'HST'
        elif rex.match('.*MDT', name):   # Mountain
            tz = 'MDT'
        elif rex.match('.*CDT', name):   # Central
            tz = 'CDT'
        elif rex.match('.*CST', name):   # Central
            tz = 'CST'
        elif rex.match('.*GMT', name):
            tz = 'GMT'
        else:
            tz = 'EDT'  # default if nothing else

        date_name = parser.parse(name, fuzzy=True, dayfirst=False, tzinfos=tzinfo)

        # t is now a PDT datetime; convert it to UTC
        date_name = date_name.astimezone(pytz.utc)
        date_dict.append((name1, name, date_name))
        time_zone.append(tz)
    except:
        try:
            date_name = parser.parse(name, fuzzy=True, dayfirst=False, tzinfos=tzinfo)
            date_name = name.astimezone(pytz.utc)
            date_dict.append((name1, name, date_name))
            time_zone.append(tz)
        except:
            # Ignore any line with more than 50 characters
            # Ignore any line with the word "the"
            # Ignore any line with two dates
            if rex.match(r'\A.*the.*\Z', name): 
                date_dict.append((name1, '', ''))
                time_zone.append('')
                return
            # Ignore line if the year appears twice
            if rex.match('\A.*(\d{4}).*(\d{4})', name): 
                date_dict.append((name1, '', ''))
                time_zone.append('')
                return
            # Ignore line if there is no number
            if not rex.match('.*\d', name): 
                date_dict.append((name1, '', ''))
                time_zone.append('')
                return
            if len(name) > 40:
                date_dict.append((name1, '', ''))
                time_zone.append('')
                return
            # if there is year in the string
            if not rex.match(r'.*\d{4}', name):
                date_dict.append((name1, '', ''))
                time_zone.append('')
                return
            exceptions.append(name)
            date_dict.append((name1, '', ''))
            time_zone.append('')
                
# Print the date in normalized form so I can spot check. 
# Then save them to a file. 
# On any date that is not valid, make it empty, tag the row and remove it from the output.csv file. 

#--------------------------------------------------------------------------------------
def create_dates_file(output_file, date_dict, time_zone):
    dates_orig = []
    dates_new = []
    dates_date = []
    date_adj = []
    timestamp = []
    exception_data = []
    print("date_adj: ", len(date_adj))
    print("timestamp: ", len(timestamp))  # Wrong number of elements

    for i, (tz_el, el) in enumerate(zip(time_zone, date_dict)):
        dct = {}
        #print("el: ", el)
        dct['el'] = el
        dates_orig.append(el[0])
        dates_new.append(el[1])
        dates_date.append(el[2])
        try:
            # subtract 5 hours to convert back to Tallahassee time

            if False:  # take daylight savings properly into account
                if tz_el[1] == 'S':  # adjust time if Savings
                    timestp -= 3600
            else:
                #print("else, el: ", el)
                #if el[2] == '':
                #    raise ValueError('el[2] does not have a timestamp attribute')
                timestp = el[2].timestamp() - 5 * 3600
            #print("dct")
            dct['timestp'] = timestp
            #print("timestp: ", timestp)
            dtime = timestampToDateTimeUTC(timestp) # many exceptions
            dct['dtime'] = dtime
            #print("dtime: ", dtime)

            # Transform time from timestamp back to UTC
            timestamp.append(timestp)
            date_adj.append(dtime)
        except:
            #print("-------------------------------------------")
            #print("   except:  ", el)
            #print("except: ", dct)
            exception_data.append(dct)
            timestamp.append(-1)  # all timestamps defined in exception
            date_adj.append(('',''))

    print(len(date_dict), len(dates_orig)) #, df.shape)
    print(len(dates_new), len(time_zone), len(date_adj))
    print("date_adj: ", len(date_adj))
    print("timestamp: ", len(timestamp))  # Wrong number of elements
    print("exception_data: ", len(exception_data))
    df1 = pd.DataFrame({'orig':dates_orig, 'new':dates_new, 'date':dates_date, 'TZ': time_zone, 'date_adj': date_adj, 'timestamp': timestamp})

    df1.to_csv(output_file, index=0)
    return date_adj, timestamp

#---------------------------------------------------------------------------
def update_output_file(df, output_file, date_adj, timestamp):
    # Add new columns: new date and time, and number of seconds since 1970
    # timestamp: seconds since 1970
    # dates_orig: original send column
    # date_adj[0]: adjusted date
    # date_adj[1]: adjusted time
    adj_date = []
    adj_time = []
    for dat_tim in date_adj:
        adj_date.append(dat_tim[0])
        adj_time.append(dat_tim[1])

    df['timestamp'] = timestamp
    df['date_sent'] = adj_date
    df['time_sent'] = adj_time

    df.to_csv(output_file, index=0)

#===========================================================================
class DateClass:
    def __init__(self):
        self.date_dict = []
        self.exceptions = []
        self.time_zone = []

        self.tzinfo = {
          "EDT": pytz.timezone('US/Eastern'),
          "CDT": pytz.timezone('US/Central'),
          "MDT": pytz.timezone('US/Mountain'),
          "PDT": pytz.timezone('US/Pacific'), 
          "HDT": pytz.timezone('US/Hawaii'), 
          "GMT": pytz.timezone('UTC'),
    
          # We will convert all times to UTC, then to seconds since 1970
          # Then we will add/subtrace one hour (3600 sec) if time is in 
          # Savings Time (as opposed to Daylight Time)
          "EST": pytz.timezone('US/Eastern'),
          "CST": pytz.timezone('US/Central'),
          "MST": pytz.timezone('US/Mountain'),
          "PST": pytz.timezone('US/Pacific'), 
          "HST": pytz.timezone('US/Hawaii'),  
         }

    def normalize_dates(self, sent_list):
        for sent in sent_list:
            self.normalize(sent)

    def normalize(self, sent):
        normalize(sent, self.exceptions, self.date_dict, self.time_zone, self.tzinfo)

    def create_dates_file(self, output_file="dates.csv"):
        self.date_adj, self.timestamp = create_dates_file(output_file, self.date_dict, self.time_zone)

    def update_output_file(self, df, output_file):
        update_output_file(df, output_file, self.date_adj, self.timestamp)


#------------------------------------------------------------------------
