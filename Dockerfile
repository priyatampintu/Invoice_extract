FROM python:3.9-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

#copy local code to container image
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./
RUN ls -la $APP_HOME/

RUN pip3 install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8503", "--server.address=0.0.0.0"]
