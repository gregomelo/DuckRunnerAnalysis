FROM python:3.12-alpine

COPY . /src

WORKDIR /src

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT [ "sh", "-c", "streamlit run app/main.py & wait" ]
