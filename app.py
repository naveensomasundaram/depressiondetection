from flask import Flask,render_template,url_for,request, json, jsonify
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle
import os
from datetime import datetime
from pytz import timezone
import os.path
from os import path
import pathlib
import re

# load the model from disk
filename = 'Model/DepressionModel.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('Model/dt_tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = re.sub('[^A-Za-z0-9]+', ' ',request.form['message'])
        username = request.form['username']

        data = [message]

        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        print(my_prediction[0])

        fmt = "%Y-%m-%d %H:%M:%S %Z%z"
        now_utc = datetime.now(timezone('UTC'))
        now_india = now_utc.astimezone(timezone('Asia/Kolkata'))
        now = now_india.strftime(fmt)

        jsonfilename = "client_data/" + username + ".json"
        file = pathlib.Path(jsonfilename)
        previous_data = []

        # reading json file..
        if file.exists():
            with open(jsonfilename) as f:
                d = json.load(f)
                previous_data = (d)


        response = { "username": username,"prediction": str(my_prediction[0]),"date": now,"text": message}
        previous_data.append(response)

        # write data into file..
        with open(jsonfilename, 'w') as f:
            json.dump(previous_data, f)

        print("completed.")
        response = { "username": username, "response": previous_data}

        # response = json.loads(response)
        return render_template('result.html',prediction = response)

if __name__ == '__main__':
    app.run(debug=True)