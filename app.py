from flask import Flask, Response, render_template
from cvzone.PoseModule import PoseDetector
import cv2

app = Flask(__name__)

# Initialize the PoseDetector
detector = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=False,
                        smoothSegmentation=True,
                        detectionCon=0.5,
                        trackCon=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, img = cap.read()
        if not success:
            break

        # Process pose detection
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)

        if lmList:
            # Process keypoints, distances, or angles as needed
            center = bboxInfo["center"]
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        # Encode the image as JPEG
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        # Yield the frame in the MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # HTML page to display the video stream

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
