from scipy.spatial import distance as dist
import playsound
import cv2

def calculate_EAR(eye):
    """ 
        Calculate the EAR (eye aspect ratio) 
        eye: 2d numpy array with shape (6,2) tương đương 6 điểm, mỗi điểm 2 tọa độ
    """
    distances_p1_p5 = dist.euclidean(eye[1], eye[5])    # nhận vào 2 1d-numpy array
    distance_p2_p4 = dist.euclidean(eye[2], eye[4])
    distance_p0_p3 = dist.euclidean(eye[0], eye[3])

    EAR = (distances_p1_p5 + distance_p2_p4) / (2 * distance_p0_p3)
    return EAR 

def sound_alarm(path_alarm):
    """ play an alarm sound """
    playsound.playsound(path_alarm)

def draw_contours(image, cnt):
    """
        Vẽ countours cho mắt để tiện theo dõi xem code của mình có ok không
        cnt: list của các tọa độ

    """
    hull = cv2.convexHull(cnt)  # xấp xỉ elip (mắt có dạng gần như vậy)
    cv2.drawContours(image, [hull], -1, (0, 255, 0), 1)