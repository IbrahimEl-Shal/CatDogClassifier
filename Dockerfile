FROM python:3.8-slim-buster

LABEL maintainer="Ibrahim El-Shal - i.hamedelshal@gmail.com"
LABEL version="0.1"
LABEL description="This is custom Docker Image for Cat_Dog_Classifier using CNN."

WORKDIR /src

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ["models", "Output", "utils", "main.py", "./"]
CMD [ "python", "./main.py" ]
