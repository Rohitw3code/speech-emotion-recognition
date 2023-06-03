import pandas as pd
import numpy as np
from tensorflow import keras
from flask import Flask, render_template, request, redirect, url_for
import librosa
import librosa.display


model = keras.models.load_model('./model//speech_model.h5')
labels = ['fear', 'angry', 'disgust', 'neutral', 'sad', 'ps', 'happy']

app = Flask(__name__, template_folder='templates', static_folder='staticFiles')

def extract_mfcc(filename):
     y, sr = librosa.load(filename, duration=3, offset=0.5)
     mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
     return mfcc



def write_to_file(username, password):
    with open('users.txt', 'a') as file:
        file.write(f'{username}:{password}\n')

def check_credentials(username, password):
    with open('users.txt', 'r') as file:
        for line in file:
            stored_username, stored_password = line.strip().split(':')
            if username == stored_username and password == stored_password:
                return True
    return False



@app.route("/home")
def home():
    return render_template("home.html")



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        write_to_file(username, password)
        return redirect(url_for('login'))
    return render_template('signup.html')



@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if check_credentials(username, password):
             return redirect(url_for('home'))
        else:
            return 'Invalid username or password'
    return render_template('login.html')



@app.route('/predict2', methods=['GET', 'POST'])
def upload_file1():
    if request.method == 'POST':
        f = request.files['file']
        print("filename ",f.filename)
        paths = [f.filename]
        pred_df = pd.DataFrame({'speech':paths})
        pred_X_mfcc = pred_df['speech'].apply(lambda x: extract_mfcc(x))
        pred_X = [x for x in pred_X_mfcc]
        pred_X = np.array(pred_X)
        pred_X = np.expand_dims(pred_X, -1)
        pred = labels[model.predict(pred_X)[0].argmax()]
        return render_template('reaction.html',mode=str(pred))


if __name__ == "__main__":
    app.run(debug=True,port=5001)
