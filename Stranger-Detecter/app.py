from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Home1.html')

@app.route('/')
def index():
    return render_template('Home2.html')
