FROM python:3.6
MAINTAINER Ayman Zeine "ayman_zeine@hms.harvard.edu"
ADD . /usr/src/app
WORKDIR /usr/src/app
EXPOSE 4000
ENV STATIC_URL /static
ENV STATIC_PATH /moseq2-app/modules/app/static
COPY ./requirements.txt /moseq2-app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "index.py"]

