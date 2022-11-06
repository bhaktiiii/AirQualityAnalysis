from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
from model import predict

app = Flask('__app__')

@app.route('/')
def index():    
    URL = 'https://aqicn.org/city/india/mumbai/kurla/'
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    print(soup.find(id='aqiwgtvalue').getText())
    curr = soup.find("div", {"id":"aqiwgtvalue"}).getText()
    live_data = { 
        'PM2.5': soup.find(id='cur_pm25').getText(),
        'PM10': soup.find(id='cur_pm10').getText(),
        'O3': soup.find(id='cur_o3').getText(),
        'NO2': soup.find(id='cur_no2').getText(),
        'SO2': soup.find(id='cur_so2').getText(),
        'CO': soup.find(id='cur_co').getText()
    }
    
    overall_prediction = predict()

    return render_template('index.html', ans = live_data, Overall = int(overall_prediction[0]))

if __name__ == "__main__":
    app.run(debug = True)