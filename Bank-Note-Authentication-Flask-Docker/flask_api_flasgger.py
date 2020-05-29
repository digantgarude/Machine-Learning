from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
from flasgger import Swagger

"""
Swagger API on : http://127.0.0.1:5000/apidocs/
"""

pickle_in = open("classifier.pkl","rb") 
classifier = pickle.load(pickle_in)


app = Flask(__name__)
Swagger(app)

@app.route("/")
def welcome():
    return "<h2><i>Swagger API on :</i> <a href='http://127.0.0.1:5000/apidocs/'>http://127.0.0.1:5000/apidocs/</a></h2>"

@app.route('/predict',methods=["GET"])
def predict():
    """Predict Individual Examples
    Enter the values ain the fields below and execute.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
      200:
        description: Output
    """
    variance = request.args.get("variance")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")
    
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    
    
    return "The prediction is : "+str(prediction)


@app.route('/predict_file',methods=["POST"])
def predict_file():
    """Predict Multiples Examples
    Upload a CSV file containing columns of variance, skewness, curtosis and entropy.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Output Values
    """
    file_ = request.files.get("file")
    df = pd.read_csv(file_)
    prediction = classifier.predict(df)
    
    
    return "The prediction for data inside file is : "+str(list(prediction))


if __name__ == "__main__":
    app.run()