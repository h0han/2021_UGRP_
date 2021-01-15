import cv2
import numpy as np

# font 등록
font = cv2.FONT_HERSHEY_SIMPLEX

# 영상에서 인식할 소스 등록
def faceDetect():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# 웹캠 활성화
    try:
        cap = cv2.VideoCapture(0)
    except:
        print('카메라 로딩 실패')
        return
    
# 얼굴을 인식하는 사각형에 대한 소스
    while True:
        ret, frame = cap.read()
        if not ret:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 2, 0, (30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3, 4, 0)
            cv2.putText(frame, 'Detected Face', (x-5, y-5), font, 0.9, (255, 255, 0), 2)

# 영상을 출력하는 소스
        cv2.imshow('frame', frame)

# space를 누르면 실행 종료
        if cv2.waitKey(1) != 255:
            break
    
    cap.release()
    cv2.destroyAllWindows()

    faceDetect()