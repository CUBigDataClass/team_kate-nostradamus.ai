FROM python:3.7.13

COPY . .

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# gunicorn
# CMD ["gunicorn", "--config", "gunicorn-cfg.py", "run:app"]
CMD gunicorn --bind 0.0.0.0:$PORT wsgi