from flask import Flask,request
import pandas as pd
import numpy as np
import pickle


pickle_in = open("classifier.pkl","rb") 
classifier = pickle.load(pickle_in)


app = Flask(__name__)


@app.route("/")
def welcome():
    return "Hello !"

@app.route('/predict')
def predict():
    variance = request.args.get("variance")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")
    
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    
    
    return "The prediction is : "+str(prediction)


@app.route('/predict_file',methods=["POST"])
def predict_file():
    file_ = request.files.get("file")
    df = pd.read_csv(file_)
    prediction = classifier.predict(df)
    
    
    return "The prediction for data inside file is : "+str(list(prediction))


if __name__ == "__main__":
    app.run()