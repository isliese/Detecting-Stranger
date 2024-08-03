from flask import Flask, render_template, redirect, url_for, request, flash, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User
import detect_mask_video  # Import the module with video stream logic
import os

app = Flask(__name__)
app.secret_key = "hgenie22"

# 현재 있는 파일의 디렉토리 절대경로
basedir = os.path.abspath(os.path.dirname(__file__))
# DB 파일 경로
dbfile = os.path.join(basedir, 'db.sqlite')

# 내가 사용 할 DB URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + dbfile
# 비지니스 로직이 끝날때 Commit 실행(DB 반영)
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
# 수정사항에 대한 TRACK
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
db.app = app

# Flask-login 설정
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# 로그인 관리자 사용자 로더
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# 홈 화면
@app.route('/')
def index():
    return render_template('Home1.html')

# 로그인 화면
@app.route('/Login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('emptycam1select'))  # 로그인 성공 시 cam1 화면으로 리다이렉트
        else:
            flash('로그인에 실패했습니다. 이메일과 비밀번호를 다시 확인해주세요.', 'danger')
    return render_template('Login.html')

# 회원가입 화면
@app.route('/SignUp', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # 이메일 중복 확인
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('이미 사용 중인 이메일입니다.', 'danger')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')  # 해시 메서드를 pbkdf2:sha256으로 변경
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('회원가입에 성공하였습니다!', 'success')
        
        login_user(new_user)
        
        return redirect(url_for('cam1'))  # 회원가입 성공 시 cam1 화면으로 리다이렉트
    return render_template('SignUp.html')

# 로그아웃
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# 카메라 선택 화면
@app.route('/CamSelect')
@login_required
def camselect():
    return render_template('CamSelect.html', username = current_user.username)

# 빈 카메라 선택 화면
@app.route('/EmptyCamSelect')
@login_required
def emptycamselect():
    return render_template('EmptyCamSelect.html')

# 빈 카메라 1 화면
@app.route('/EmptyCam1Select')
@login_required
def emptycam1select():
    return render_template('EmptyCam1Select.html')

# 카메라 1 화면
@app.route('/Cam1')
@login_required
def cam1():
    return render_template('Cam1.html', username = current_user.username)

# 지인 등록 화면
@app.route('/RegisteredCard')
@login_required
def registeredcard():
    return render_template('RegisteredCard.html', username = current_user.username)

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(detect_mask_video.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 데이터베이스 파일 삭제 전 연결 종료
    if os.path.exists(dbfile):
        try:
            os.remove(dbfile)
            print("Database file deleted.")
        except PermissionError as e:
            print(f"PermissionError: {e}")
    
    # Flask 앱 초기화
    with app.app_context():
        db.create_all()
        print("Database initialized.")
    
    # Flask 앱 실행
    app.run(debug=True, port=5001)
