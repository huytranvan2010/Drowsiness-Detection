# Drowsiness-Detection
Trong bài này chúng ta đi tìm hiểu một ứng dụng của facial landmarks là **drowsiness detection** - phát hiện ngủ gật. Ứng dụng này có thể hỗ trợ các bác tài xế khi lái xe, đặc biệt khi lái xe trong thời gian dài. 

Ý tưởng cho ứng dụng này là sử dụng metric **eye aspect ratio (EAR)** được giới thiệu trong bài báo [Real-Time Eye Blink Detection Using Facial Landmarks.](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)

Cùng nhớ lại các vị trí của facial landmarks được tạo ra từ facial landmark detector trong thư viện dlib.
<img src="https://camo.githubusercontent.com/4d074bd6665655e2b8267d665a0cf72d5002ff2eecac61ecf3c516a6b6605880/68747470733a2f2f7777772e7079696d6167657365617263682e636f6d2f77702d636f6e74656e742f75706c6f6164732f323031372f30342f66616369616c5f6c616e646d61726b735f36386d61726b75702d373638783631392e6a7067">

Đối với mỗi mắt chúng ta có 6 facial landmarks. 
<img src="https://www.pyimagesearch.com/wp-content/uploads/2017/04/blink_detection_plot.jpg">

*Facial landmarks khi mắt mở và mắt nhắm*
Lưu ý: các chỉ số trên hình không khớp với chỉ số cho facial landmarks của mắt nhận được từ dlib.
Eye aspect ratio (EAR) được định nghĩa như sau:
$$ EAR = \frac{\left\|p_2 - p_6 \right\| + \left\|p_3 - p_5 \right\|}{2\left\|p_1 - p_4 \right\|} $$

$\left\| \right\|$ - Euclidean distance giữa 2 điểm (có thể build lại hoặc dùng thư viện cho nhanh như `scipy`).
Nhận thấy khi mắt mở giá trị của EAR gần như không đổi quanh một ngưỡng nào đó (trên hình khoảng 0.25), khi mắt nhắm EAR giảm xuống (cần set ngưỡng). Nếu mắt nhắm trong thời gian đủ lâu chúng ta sẽ phát cảnh báo cho tài xế. Thêm nữa do mỗi người có hai mắt nên ta sẽ lấy giá trị trung bình của 2 mắt cho EAR (người bị thiếu một mắt chắc không đủ điều kiện lái xe). Cần test cả mode cho trường hợp lái xe đeo kính (đeo kính râm đen thì khó).

Mình xin tóm lại các bước chính như sau:
* **Bước 1:** Detect khuôn mặt
* **Bước 2:** Xác định facial landmarks dựa trên khuôn mặt đã detect được ở bước 1
* **Bước 3:** Xác định EAR (eye aspect ration) cho 2 mắt (lấy gá trị trung bình)
* **Bước 4:** Kiểm tra điều kiệm cảnh cáo nếu có.

Source code các bạn có thể xem tại [github-huytranvan2010](https://github.com/huytranvan2010/Drowsiness-Detection), mình có comment rất chi tiết các dòng lệnh.


