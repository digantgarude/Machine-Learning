# Bank note authentication usecase on Flask and Dockers Testing

[Original Dataset Link](https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data)

### Status : Work in Progress ;)

## Installation

```bash
pip install pandas
pip install pickle
pip install flask
pip install flasgger
```

## To run python code
```bash
python flask_api_flasgger.py
```

## To access the swagger api on browser. (Default server port is 5000 for flask)
```
http://127.0.0.1:5000/apidocs/
```

### Docker Steps

1. Create *requirements.txt*

```bash
pip freeze > requirements.txt
```

2. Create *Dockerfile*

```
FROM continuumio/anaconda3:4.4.0
COPY . /usr/app
EXPOSE 5000
WORKDIR /usr/app
RUN pip install -r requirements.txt
CMD python flask_api_flasgger.py
```

3. To build the Docker image

```bash 
sudo docker build -t note_forgery_api .
```

4. To run the Docker image.

```bash
sudo docker run -p 8000:8000 note_forgery_api
```

5. Check which dockers are running.

```bash
sudo docker ps
```



[Krish Naik Sir's Dockers Tutorial](https://www.youtube.com/watch?v=hTacGMfL8lc)