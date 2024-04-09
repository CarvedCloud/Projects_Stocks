"""
Created on Tue Jul 18 13:42:18 2023

@author: srini
"""

#Importing Modules/Libraries
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import requests               #Used for http/internet requests
import xlsxwriter
import math 
import yfinance as yfin
import time

s=1
c=0

#from secrets import IEX_CLOUD_API_TOKEN
#import pyEX

'''
API Token (IEX Cloud):     pk_6a949575a5d34bfd8a44d9a15ea9b91d
Support: https://iexcloud.io/docs/api/#support


#API Call Test
symbol = 'ASHOKLEY'
api_url = f'https://cloud.iexapis.com/stable/stock/{symbol}/stats?token=pk_6a949575a5d34bfd8a44d9a15ea9b91d'
data = requests.get(api_url).json()
'''


#Import list of stocks used
stocks = pd.read_csv('C:\\Programming\\Python\\Projects\\nifty500list.csv')
stock_codes=stocks.filter(["ISIN Code"])
stock_codes_list=stock_codes.values

hqm_columns = [
                'ISIN Code',
                'Name',
                'Industry',
                'Momentum Score',
                'Price To Earning Ratio',
                'Current Price', 
                'Number of Shares to Buy', 
                'One-Year Price Return', 
                'One-Year Return Percentage',
                'Six-Month Price Return',
                'Six-Month Return Percentage',
                'Three-Month Price Return',
                'Three-Month Return Percentage',
                'One-Month Price Return',
                'One-Month Return Percentage'
                ]

hqm_dataframe = pd.DataFrame(columns = hqm_columns)

for i in stock_codes_list:
    stock_code_temp = yfin.Ticker(i[0])
    status = 0
    
    #1 year
    t=0
    while True:
        data = stock_code_temp.history(period="1y")
        if len(data)!=0:
            close_data = data.filter(['Close'])
            close_dataset = close_data.values
            _1y_price_return_value = close_dataset[-1]-close_dataset[0]
            _1y_price_return_percent = _1y_price_return_value/close_dataset[0]
            tempvar1=_1y_price_return_value/((close_dataset[-1]+close_dataset[0])/2)
            data.iloc[0:0]          #Empty Dataframe called data    
            break
        
        stock_code_temp = yfin.Ticker(i[0])
        t+=1
        if t>10:
            c+=1
            status = 1
            break
        time.sleep(1)
        
    #6 months
    t=0
    while True:
        data = stock_code_temp.history(period="6mo")
        if len(data)!=0:
            close_data = data.filter(['Close'])
            close_dataset = close_data.values  
            _6mo_price_return_value = close_dataset[-1]-close_dataset[0]
            _6mo_price_return_percent = _6mo_price_return_value/close_dataset[0]
            tempvar2=_6mo_price_return_value/((close_dataset[-1]+close_dataset[0])/2)
            data.iloc[0:0]
            break
        
        stock_code_temp = yfin.Ticker(i[0])
        t+=1
        if t>0:
            break
    
    #3 months
    t=0
    while True:
        data = stock_code_temp.history(period="3mo")
        if len(data)!=0:
            close_data = data.filter(['Close'])
            close_dataset = close_data.values  
            _3mo_price_return_value = close_dataset[-1]-close_dataset[0]
            _3mo_price_return_percent = _3mo_price_return_value/close_dataset[0]
            tempvar3=_3mo_price_return_value/((close_dataset[-1]+close_dataset[0])/2)
            data.iloc[0:0]
            break
        stock_code_temp = yfin.Ticker(i[0])
        t+=1
        if t>0:
            break
    
    #1 months
    t=0
    while True:
        data = stock_code_temp.history(period="1mo")
        if len(data)!=0:
            close_data = data.filter(['Close'])
            close_dataset = close_data.values  
            _1mo_price_return_value = close_dataset[-1]-close_dataset[0]
            _1mo_price_return_percent = _1mo_price_return_value/close_dataset[0]
            tempvar4=_1mo_price_return_value/((close_dataset[-1]+close_dataset[0])/2)
            data.iloc[0:0]
            break            
        stock_code_temp = yfin.Ticker(i[0])
        t+=1
        if t>0:
            break
    
    #Calculating Scores
    momentum_score = _1y_price_return_percent*100 + _6mo_price_return_percent*100 + _3mo_price_return_percent*100 + _1mo_price_return_percent*100
    price_earning_ratio = (tempvar1 + tempvar2 + tempvar3 + tempvar4)/4
    
    row_num = stocks[stocks['ISIN Code'] == i[0]].index
    name=stocks['Company Name'].loc[stocks.index[row_num[0]]]
    industry=stocks['Industry'].loc[stocks.index[row_num[0]]]

    
    #Appending values into dataframe
    tempdf = {'ISIN Code':i[0],
              'Name':name,
              'Industry':industry,
              'Current Price':float(close_dataset[-1][0]),
              'One-Year Price Return':float(_1y_price_return_value[0]),
              'One-Year Return Percentage':float(_1y_price_return_percent[0]),
              'Six-Month Price Return':float(_6mo_price_return_value[0]),
              'Six-Month Return Percentage':float(_6mo_price_return_percent[0]),
              'Three-Month Price Return':float(_3mo_price_return_value[0]),
              'Three-Month Return Percentage':float(_3mo_price_return_percent[0]),
              'One-Month Price Return':float(_1mo_price_return_value[0]),
              'One-Month Return Percentage':float(_1mo_price_return_percent[0]),
              'Momentum Score':float(momentum_score[0]),
              'Price To Earning Ratio':float(price_earning_ratio[0])
              }
    if status == 0:
        hqm_dataframe = hqm_dataframe.append(tempdf, ignore_index = True)
    
    print("No. Of Stocks Analysed: ",s-c)
    print("No. Of Failed Entries: ", c)
    print("")
    s+=1

'''
#Saving Data in Excel (Standard)
hqm_dataframe.to_csv('Stock_Analysis.csv')      
#In the percent calculations for every time frame, manually multiply by hundred as it is done automatically in the fancy csv save method
'''
    
#Saving Data in Excel (Fancy Way)
writer = pd.ExcelWriter('Stock_Analysis.xlsx', engine='xlsxwriter')
hqm_dataframe.to_excel(writer, sheet_name='Momentum Strategy', index = False)

background_color = '#0a0a23'
font_color = '#ffffff'

string_template = writer.book.add_format(
        {
            'font_color': font_color,
            'bg_color': background_color,
            'border': 1
        }
    )

dollar_template = writer.book.add_format(
        {
            'num_format':'₹0.000',
            'font_color': font_color,
            'bg_color': background_color,
            'border': 1
        }
    )

integer_template = writer.book.add_format(
        {
            'num_format':'0.000',
            'font_color': font_color,
            'bg_color': background_color,
            'border': 1
        }
    )

percent_template = writer.book.add_format(
        {
            'num_format':'0.00%',
            'font_color': font_color,
            'bg_color': background_color,
            'border': 1
        }
    )
column_formats = { 
                    'A': ['ISIN Code', string_template],
                    'B': ['Name', string_template],
                    'C': ['Industry', string_template],
                    'D': ['Momentum Score', integer_template],
                    'E': ['Price To Earning Ratio', integer_template],
                    'F': ['Current Price', dollar_template],
                    'G': ['Number of Shares to Buy', integer_template],
                    'H': ['One-Year Price Return', dollar_template],
                    'I': ['One-Year Return Percentage', percent_template],
                    'J': ['Six-Month Price Return', dollar_template],
                    'K': ['Six-Month Return Percentage', percent_template],
                    'L': ['Three-Month Price Return', dollar_template],
                    'M': ['Three-Month Return Percentage', percent_template],
                    'N': ['One-Month Price Return', dollar_template],
                    'O': ['One-Month Return Percentage', percent_template],
                    
                    }

for column in column_formats.keys():
    writer.sheets['Momentum Strategy'].set_column(f'{column}:{column}', 20, column_formats[column][1])
    writer.sheets['Momentum Strategy'].write(f'{column}1', column_formats[column][0], string_template)

writer.save()
