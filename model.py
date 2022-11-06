import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
import requests
from bs4 import BeautifulSoup

seed = 7
np.random.seed(seed)
const_loss=20

def to_supervised(data,dropNa = True,lag = 1):
    df = pd.DataFrame(data)
    column = []
    column.append(df)
    for i in range(1,lag+1):
        column.append(df.shift(-i))
    df = pd.concat(column,axis=1)
    df.dropna(inplace = True)
    features = data.shape[1]
    df = df.values
    supervised_data = df[:,:features*lag]
    supervised_data = np.column_stack( [supervised_data, df[:,features*lag]])
    return supervised_data

def predict():
    model = keras.models.load_model('./static/aqi_model.h5')

    d1 = pd.read_csv("./static/kurla,-mumbai-air-quality.csv")
    d1 = d1.sort_values(by="date")
    d1= d1.rename(columns = {" pm25": "pm25", 
                            " pm10":"pm10", 
                            " o3": "o3",
                            ' no2' : 'no2',
                            ' so2' : 'so2',
                            ' co' : 'co'})
    features=list(d1.columns)    
    d1['pm25'] = pd.to_numeric(d1['pm25'],errors='coerce')
    d1['pm10'] = pd.to_numeric(d1['pm10'],errors='coerce')
    d1['o3'] = pd.to_numeric(d1['o3'],errors='coerce')
    d1['no2'] = pd.to_numeric(d1['no2'],errors='coerce')
    d1['so2'] = pd.to_numeric(d1['so2'],errors='coerce')
    d1['co'] = pd.to_numeric(d1['co'],errors='coerce')

    URL = 'https://aqicn.org/city/india/mumbai/kurla/'
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    pm_25=float(soup.find(id='cur_pm25').getText())
    pm_10=float(soup.find(id='cur_pm10').getText())
    cur_o3=float(soup.find(id='cur_o3').getText())
    cur_no2=float(soup.find(id='cur_no2').getText())
    cur_co=float(soup.find(id='cur_co').getText())
    cur_so2=float(soup.find(id='cur_so2').getText())
    
    lived_data = np.array([pm_25,pm_10,cur_o3,cur_no2,cur_so2,cur_co])

    live_data={"date":'2022/2/10',"pm25":pm_25,'pm10':pm_10,'o3':cur_o3,'no2':cur_no2,'so2':cur_so2,'co':cur_co}
    d1=d1.append(live_data,ignore_index = True)

    for label,content in d1.items():
        if label != 'date':
            d1[label]=d1[label].fillna(d1[label].mean())
            
    d1['date']=pd.to_datetime(d1['date'])
    d1.set_index(d1['date'], inplace = True) 
    d1=d1.drop(['date'],axis=1)

    data=d1
    # print(data.head())
    # preprocess the wind direction with label encoding
    from sklearn.preprocessing import LabelEncoder
    values = data.values
    encoder = LabelEncoder()

    values[:,4] = encoder.fit_transform(values[:,4])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    
    timeSteps = 2
    supervised = to_supervised(scaled,lag=timeSteps)
    
    pd.DataFrame(supervised)
    features = data.shape[1]
    train_hours = 365
    X = supervised[:,:features*timeSteps]
    y = supervised[:,features*timeSteps]

    val=X[-1].reshape(1,2,6)
    y_pred = model.predict(val)
    r = val.reshape(val.shape[0],val.shape[2]*val.shape[1])
    inv_new = np.concatenate( (y_pred, r[:,-5:] ) , axis =1)
    inv_new = scaler.inverse_transform(inv_new)
    final_pred = inv_new[:,0]
    final_pred-const_loss#for loss
    return final_pred