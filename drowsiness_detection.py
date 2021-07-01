# USAGE
# python drowsiness_detection.py --shape_predictor shape_predictor_68_face_landmarks.dat --alarm preview.mp3
import imutils
from imutils import face_utils
import argparse
import dlib
import cv2
from hammiu import calculate_EAR, sound_alarm, draw_contours
from hammiu import config
from threading import Thread 

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--shape_predictor", required=True, help="path to the facial landmarks predictor")
parser.add_argument("-a", "--alarm", required=True, help="path to the alarm file")
parser.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(parser.parse_args())

# tạo biến đếm số lần
total = 0
# lưu có bật tín hiện cảnh báo hay không
alarm_on = False  

# Lấy chỉ số của facial landmarks cho left eye and right eye
# Hoàn toàn có thể bỏ qua, vì left eye [36:42], right eye [42:48]
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Đầu tiên vẫn phải phát hiện khuôn mặt trước, khởi tạo dlib's face detector từ dlib, dựa trên HOG + Linear SVM
detector = dlib.get_frontal_face_detector()
# Tạo facial landmark predictor
predictor = dlib.shape_predictor(args["shape_predictor"])

video = cv2.VideoCapture(args["webcam"])

while True:
    ret, frame = video.read()

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)   # tham số thứ 2 là số lần upsample image (tăng kích thước ảnh để phát hiện mặt nhỏ)

    for rect in rects:
        # xác định facial landmarks
        shape = predictor(gray, rect)
        # chuyển về 2d numpy array có shape (68, 2)
        shape = face_utils.shape_to_np(shape)

        # Lấy các tọa độ facial landmarks cho mắt trái, EAR cho mắt trái
        left_eye = shape[left_start:left_end]
        left_EAR = calculate_EAR(left_eye)

        # Lấy các tọa độ facial landmarks cho mắt phải, EAR cho mắt phải
        right_eye = shape[right_start:right_end]
        right_EAR = calculate_EAR(right_eye)

        EAR = (left_EAR + right_EAR) / 2.
        cv2.putText(frame, "EAR: {:.2f}".format(EAR) ,(300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        # vẽ contour xung quanh hai mắt để tiện theo dõi
        draw_contours(frame, left_eye)
        draw_contours(frame, right_eye)

        """ Bắt đầu kiểm tra xem có dấu hiệu buồn ngủ không """
        if EAR > config.EAR_THRESHOLD:  # mở mắt
            total = 0
            alarm_on = False
            cv2.putText(frame, "Eyes open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        else: # nhắm mắt
            total += 1

            if total > config.EAR_CONSEC_FRAMES:
                """ 
                    Đang cảnh báo alarm_on=True thì cảnh báo tiếp
                    Chưa cảnh báo alarm_on=False thì bắt đầu cảnh báo, đổi alarm_on=True    
                """ 
                # Chưa bật cảnh báo thì bật
                if not alarm_on:
                    alarm_on = True
                    t = Thread(target=sound_alarm, args=(args["alarm"],))
                    t.deamon = True
                    t.start()
                    sound_alarm(args["alarm"])

                cv2.putText(frame, "Drowsiness detect" ,(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()








