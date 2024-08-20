from flask import Flask, render_template, redirect, url_for, request, Response, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User
import detect_mask_video  # Import the module with video stream logic
import os
import base64
import datetime
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import subprocess
import threading
import time
from imutils.video import VideoStream

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

# 전역 VideoStream 객체와 관련 변수
video_stream = None
video_stream_thread = None

# 비디오 스트리밍 시작
def start_video_stream():
    global video_stream, video_stream_thread
    if video_stream is None:
        video_stream = VideoStream(src=0).start()
        video_stream_thread = threading.Thread(target=video_stream.update)
        video_stream_thread.start()

# 비디오 스트리밍 중지
def stop_video_stream():
    global video_stream, video_stream_thread
    if video_stream is not None:
        video_stream.stop()
        video_stream = None
        if video_stream_thread is not None:
            video_stream_thread.join()
            video_stream_thread = None

# 서버 코드에 추가할 엔드포인트
@app.route('/status_update')
@login_required
def status_update():
    global last_warning
    return jsonify({"message": last_warning})

last_warning = None  # 전역 변수로 선언

def monitor_output():
    process = subprocess.Popen(
        ['python3', 'detect_mask_video.py'],  # 실행할 스크립트
        stdout=subprocess.PIPE,               # 표준 출력을 파이프에 연결
        stderr=subprocess.PIPE,               # 표준 오류를 파이프에 연결
        text=True                             # 텍스트 모드로 출력 읽기
    )
    
    while True:
        output = process.stdout.readline()  # 표준 출력에서 한 줄 읽기
        if output:
            output = output.strip()  # 공백 제거
            if "Warning: '미등록자'가 50번 연속 감지되었습니다!" in output:
                print(output)  # 또는 다른 처리 방법
        if process.poll() is not None:  # 서브프로세스가 종료되었는지 확인
            break


# 홈 화면
@app.route('/')
def index():
    return render_template('Home1.html')

# 로그인 화면
@app.route('/Login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    elif request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'Invalid request, expecting JSON data.'}), 400
        email = data.get('email')
        password = data.get('password')
    
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            response = {
                'success': True,
                'message': '로그인에 성공했습니다.',
                'redirect_url': url_for('cam1')
            }
        else:
            response = {
                'success': False,
                'message': '로그인에 실패했습니다. 이메일과 비밀번호를 다시 확인해주세요.'
            }    
        return jsonify(response)

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
            return render_template('SignUp.html', email_exists=True)

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')  # 해시 메서드를 pbkdf2:sha256으로 변경
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        
        return render_template('SignUp.html', success=True, username=username)
    
    return render_template('SignUp.html')

# 로그아웃
@app.route('/logout', methods=['POST'])
@login_required
def logout():
    stop_video_stream()
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
    start_video_stream()
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

# 지인 얼굴 등록 화면 (Capture.html 크게 띄울 화면)
@app.route('/Capture')
@login_required
def capture():
    name = request.args.get('name', '')
    return render_template('Capture.html', username = current_user.username, name=name)

# 얼굴 등록 페이지 시작
@app.route('/start_capture/<name>')
@login_required
def start_capture(name):
    # 전달받은 이름으로 폴더 이름 생성
    folder_name = name if name else "New_User"
    folder_path = os.path.join('dataset', folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    return render_template('capture_page.html', folder_name=folder_name)

@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    data = request.get_json()
    image_data = data.get('image')
    folder_name = data.get('folder')
    
    if image_data and folder_name:
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        
        # 이미지 처리
        image = Image.open(BytesIO(image_bytes))
        open_cv_image = np.array(image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        
        # 얼굴 인식
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # 얼굴 부분만 저장
        face_count = 0
        for (x, y, w, h) in faces:
            face = open_cv_image[y:y+h, x:x+w]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = os.path.join('dataset', folder_name, f'face_{timestamp}.jpg')
            cv2.imwrite(save_path, face)
            face_count += 1

        return jsonify({'success': True, 'message': f'{face_count} faces saved to {folder_name}'})
    
    return jsonify({'success': False, 'message': 'Invalid data.'})

@app.route('/train_model', methods=['POST'])
@login_required
def train_model_route():
    # 학습 상태를 전역 변수로 관리
    global training_completed
    training_completed = False
    
    # 별도의 스레드에서 학습 시작
    threading.Thread(target=train_model).start()
    
    return jsonify({"success": True, "message": "모델 학습이 시작되었습니다."})

def train_model():
    global training_completed
    # 학습 스크립트를 호출 (예: subprocess를 사용하여 외부 스크립트 실행)
    subprocess.call(['python', 'train_mask_detector.py'])
    training_completed = True

@app.route('/training_status', methods=['POST'])
@login_required
def training_status():
    global training_completed
    return jsonify({"status": "completed" if training_completed else "in_progress"})

@app.route('/confirm_training_complete', methods=['POST'])
@login_required
def confirm_training_complete():
    # 학습 완료 후 확인 버튼 클릭 시 리디렉션
    return redirect(url_for('registeredcard'))

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
    
    # 모니터링 스레드 시작
    monitoring_thread = threading.Thread(target=monitor_output)
    monitoring_thread.start()
    
    # Flask 앱 실행
    app.run(debug=True, port=5001)
