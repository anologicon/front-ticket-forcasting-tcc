FROM python:3.7

EXPOSE 8501

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY ./webapp ./webapp

