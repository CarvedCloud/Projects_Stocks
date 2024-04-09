"""
Created on Sun Jul  9 21:25:57 2023

@author: srini
"""

#Description: This program uses an artificial recurrent neural netork called Long Short Term Memory (LSTM) to predict the closing stock price of a corporation using the past 60 days stock price.

'''
Note: Whenever changing between daily and weekly data, change df to fdf and vice versa
'''

#Import Libraries
import math
from pandas_datareader import data as pdr
import yfinance as yfin
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import date, timedelta
plt.style.use('fivethirtyeight')
n=0
s=0
fdf=pd.DataFrame()


#Define Functions
def get_days(start, day_index, end=None):
    # set the start as the next valid day
    start += timedelta(days=(day_index - start.weekday()) % 7)
    week = timedelta(days=7)
    while end and start < end or not end:
        yield start
        start += week   
        global s
        s+=1

#Get the stock quote
sdate = date(2000,1,1)           # start date
edate = date(2023,7,27)          # end date
stock_name = 'RELIANCE.NS'
error=1000000000000000000000000000000

thursday_generator = get_days((sdate), 3, (edate))
tl = list(thursday_generator)

friday_generator = get_days((sdate), 4, (edate))
fl = list(friday_generator)


#Daily Data 
# yfin.pdr_override()
# df = pdr.get_data_yahoo(stock_name,start=sdate,end=edate)
# df = df[:-2]


#Weekly Data
yfin.pdr_override()
for i in range(0,len(tl)):    
    sd = str(tl[i])
    df = pdr.get_data_yahoo(stock_name,start=tl[i],end=fl[i])
    temp = [fdf,df]
    fdf = pd.concat(temp)
    

# Convert to weekly data (Old Method, maybe wrong)
# agg_dict = {'Open': 'first',
#           'High': 'max',
#           'Low': 'min',
#           'Close': 'last',
#           'Adj Close': 'last',
#           'Volume': 'mean'}

# # resampled dataframe
# # 'W' means weekly aggregation
# df = df.resample('W').agg(agg_dict)


#show the data
print("")
print("Stock: ",stock_name)
print(fdf.head(7))
print('---------')
print(fdf.tail(7))


#Get number of rows and columns in the data set
print("\nNumber of rows and columns: ",fdf.shape)


'''
#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price INR', fontsize=18)
plt.show()
'''


#Create a new dataframe with only the 'Close" column
data = fdf.filter(['Close'])
#Convert the data frame to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset)*0.8)
print("\nTraining data length: ",training_data_len)        #80% of actual length


#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
#print("\nScaled data: ")
#print(scaled_data)


#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len , :]
#Split the data into x_train and y_train data sets
x_train = []
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<=60:
        print("X trained data: ")
        print(x_train)
        print("Y trained data: ")
        print(y_train)          #Contains 61st value that we want our model to predict 
        print("")
        

#Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_train.shape


while True:
    n+=1
    #Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(50,return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))



    #Compile model
    model.compile(optimizer='adam',loss='mean_squared_error')
    
    #Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)


    #Create the testing data set
    #Create a new array containing scaled values
    test_data = scaled_data[training_data_len - 60: , :]
    #Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    


    #Convert the data to a numpy array
    x_test = np.array(x_test)



    #Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))



    #Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)


    #Get the root mean squared error (RMSE)
    rmse = np.sqrt( np.mean(predictions - y_test)**2)
    if rmse<error:
        break
print("")
print("Root Mean Squared Error: ",rmse)
print("Training Cycles: ",n)
print("")


#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title(stock_name)
plt.xlabel('date', fontsize=18)
plt.ylabel('Close Price INR', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train', 'Val','Predictions'], loc='lower right')
plt.show()


#Show the valid and predicted prices
print(valid)
print("")
print("")


#Get the quote
stock_quote = pdr.get_data_yahoo(stock_name,start=sdate,end=edate)
#Create a new dataframe
new_df = stock_quote.filter(['Close'])
#Get the last 60 days closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test=[]
#Append the past 60 days
X_test.append(last_60_days_scaled)
#Convert the X-test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print("Predicted Price: ",pred_price)
print("")
print("")













