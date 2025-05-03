from flask import Flask, render_template, Response, jsonify
from yolovideo import generate_frames

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    # 공 위치, 판정 여부 하드코딩 예시
    detection = {
        "ball": {"x": 320, "y": 250},
        "is_strike": True
    }
    return jsonify(detection)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
