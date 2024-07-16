from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Home1.html')

@app.route('/Home2')
def home2():
    return render_template('Home2.html')

@app.route('/Login')
def login():
    return render_template('Login.html')

@app.route('/SignUp')
def signup():
    return render_template('SignUp.html')

@app.route('/EmptyCamSelect')
def emptycamselect():
    return render_template('EmptyCamSelect.html')

@app.route('/CamSelect')
def camselect():
    return render_template('CamSelect.html')

@app.route('/EmptyCam1Select')
def emptycam1select():
    return render_template('EmptyCam1Select.html')

@app.route('/EmptyCam2Select')
def emptycam2select():
    return render_template('EmptyCam2Select.html')

@app.route('/Cam2')
def cam2():
    return render_template('Cam2.html')

@app.route('/RegisteredCard')
def registeredcard():
    return render_template('RegisteredCard.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
