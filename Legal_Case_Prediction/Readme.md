## Install requirements (Use python 3.11)
pip install -r /path/to/requirements.txt

## setup environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

## Run
flask run

(or)

python3 -m venv path/to/venv
source path/to/venv/bin/activate
to run : sudo python3 app.py
It will start running on localhost:5000

## Steps to predict
User needs to select the model type, and atleast provide First party, Second party and Facts. Then click on Predict button. 

