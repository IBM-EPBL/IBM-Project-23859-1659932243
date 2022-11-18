
import pickle

import numpy as np
from flask import Flask, render_template, request


filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = request.form.get('sex')
        sex=int(sex)
        cp = request.form.get('cp')
        cp=int(cp)
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        fbs=int(fbs)
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        exang=int(exang)
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        slope=int(slope)
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        thal=int(thal)
        
        
        lst1=[[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]
        print(lst1)
        
        my_prediction = model.predict(lst1)
        
        return render_template('result.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run(debug=False)

