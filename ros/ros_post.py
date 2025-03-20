#! /usr/bin/env python3
# encoding: utf-8

import rospy
import smbus
import cv2
import pyzbar.pyzbar as pyzbar
import numpy as np
import threading

from time import sleep
from jetbotmini_msgs.msg import *
from jetbotmini_msgs.srv import *
from std_msgs.msg import String

bus = smbus.SMBus(1)
ADDRESS = 0x1B

def gstreamer_pipeline(
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# 카메라 Thread
class CameraStream:
    def __init__(self):
        self.capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        self.grabbed, self.frame = self.capture.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            grabbed, frame = self.capture.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame if grabbed else None

    def read(self):
        with self.lock:
            return self.frame.copy() if self.grabbed else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.capture.release()

# ROS 노드 클래스
class PostRobot:
    def __init__(self):
	    # 노드 설정 (기본 노드 사용)
        rospy.init_node("post_robot", anonymous=False)

        # ROS 종료 (rosShutDown 함수 사용)
        rospy.on_shutdown(self.rosShutDown)

        # 카메라 스트림 시작
        self.camera = CameraStream()
        self.shared_frame = None
        self.lock = threading.Lock()
        self.running = False

        # 배터리 확인용 Publisher
        self.volPublisher = rospy.Publisher("/voltage", Battery, queue_size=10)

        # 배터리 전압을 주기적으로 퍼블리시하기 위한 Timer 설정
        self.timer = rospy.Timer(rospy.Duration(20), self.battery_callback)

        # 웹 페이지에서 명령 수신 (go, stop)
        self.command_sub = rospy.Subscriber("/robot_state", String, self.robotMove)

        # 웹 페이지에서 선택한 택배 도착 구역 토픽 수신
        self.place_sub = rospy.Subscriber("/robot_place", String, self.updatePostPlace)        

        # 배송 상태를 웹으로 전송할 Publisher
        self.delivery_status_pub = rospy.Publisher("/delivery_status", String, queue_size=10)

        # QR 코드 인식에 필요한 목적지 정보
        self.post_place = []

        # 라인 트레이싱 관련 속성 (속도, 회전)
        self.speed_value = 0.34  # 상수로 정의된 속도
        self.turn_value = 0.25   # 상수로 정의된 회전값

        # 라인 트레이싱 활성화 상태
        self.line_tracing_active = True
    
    # ROS 노드 종료
    def rosShutDown(self):
        self.running = False  # 안전한 종료 보장
        self.camera.stop()
        bus.write_i2c_block_data(ADDRESS, 0x01, [1, 0, 1, 0])
        sleep(0.01)

    # 배터리 전압을 퍼블리시하는 함수
    def battery_callback(self, event):
        AD_value = bus.read_i2c_block_data(ADDRESS, 0x00, 2)
        voltage = ((AD_value[0] << 8) + AD_value[1]) * 13.3 / 1023.0
        battery = Battery()
        battery.Voltage = voltage
        self.volPublisher.publish(battery)

    # 전달받은 QR코드 정보 저장
    def updatePostPlace(self, msg):
        if len(msg.data) > 0:
            self.post_place = msg.data.split(",")
        self.post_place.append("home")
        rospy.loginfo(f"{self.post_place}")
    
    # 라인 트레이싱
    def lineTracing(self):
        while self.running and not rospy.is_shutdown():
            # QR 코드를 감지 했을 경우 일시 정지
            if not self.line_tracing_active:                
                bus.write_i2c_block_data(ADDRESS, 0x01, [1, 0, 1, 0])
                sleep(0.01)
                continue

            with self.lock:
                if self.shared_frame is not None:
                    frame = self.shared_frame.copy()
                else:
                    sleep(0.01)  # CPU 점유율 줄이기
                    continue
            
            # 프레임 하단 절반만 사용
            height, width, _ = frame.shape
            lower_half = frame[height // 2:, :].copy()

            # HSV 변환을 위한 색 범위            
            color_lower = np.array([100, 45, 46])    # 최소값
            color_upper = np.array([124, 255, 255])  # 최대값

            # 블러 처리
            lower_half = cv2.GaussianBlur(lower_half, (5, 5), 0)

            # HSV 변환
            hsv = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV)
            hsv.astype(np.uint8)

            # 색상 범위 내에서 마스크 생성
            mask = cv2.inRange(hsv, color_lower, color_upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            
            # 윤곽선 감지
            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 속도 계산
            if len(cnts) > 0:
                cnt = max(cnts, key=cv2.contourArea)
                (color_x, color_y), color_radius = cv2.minEnclosingCircle(cnt)

                if color_radius > 30:  # 일정 크기 이상의 원만 고려
                    # 색상이 검출된 영역을 원본 `frame` 좌표에 맞게 변환
                    color_x = int(color_x)
                    color_y = int(color_y + height // 2)  # 하단 절반에서 찾았으므로 원래 위치로 보정

                    # 원본 이미지에 원을 그림
                    cv2.circle(frame, (color_x, color_y), int(color_radius), (255, 0, 255), 2)

                    # 중심 기준 오차 계산 (width 활용)
                    center_offset = (width // 2 - color_x) / (width // 2)
                    left_speed = int((self.speed_value - self.turn_value * center_offset) * 255)
                    right_speed = int((self.speed_value + self.turn_value * center_offset) * 255)

                    # 속도 제한 (0 ~ 255)
                    left_speed = max(0, min(255, left_speed))
                    right_speed = max(0, min(255, right_speed))

                    # 모터 제어 전송
                    try:
                        bus.write_i2c_block_data(ADDRESS, 0x01, [1, left_speed, 1, right_speed])  # 전진 모션
                    except Exception as e:
                        rospy.logerr(f"Failed to send motor command: {e}")

    # QR 코드 인식률을 높이기 위한 원근 변환
    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        morph_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(morph_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)

            # 원본 영상에서 QR 코드가 있는 네 모서리 좌표 (사다리꼴 형태)
            src_pts = np.float32([
                [w * 0.3, h * 0.2],     # 왼쪽 위
                [w * 0.7, h * 0.2],     # 오른쪽 위
                [w * 0.85, h * 0.85],   # 오른쪽 아래
                [w * 0.15, h * 0.85]    # 왼쪽 아래
            ])

            # 변환 후 원하는 정사각형 좌표 (정면 뷰)
            dst_pts = np.float32([
                [10, 10],               # 왼쪽 위
                [w - 10, 10],           # 오른쪽 위
                [w - 10, h - 10],       # 오른쪽 아래
                [0, h - 10]             # 왼쪽 아래
            ])

            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            corrected_frame = cv2.warpPerspective(frame, matrix, (640, 480))

            return corrected_frame
        else:
            return None

    # QR 코드 인식
    def detectedQRcode(self):
        frame_counter = 0  # QR 코드 인식할 프레임
        while self.running and not rospy.is_shutdown():
            with self.lock:
                if self.shared_frame is not None:
                    frame = self.shared_frame.copy()
                else:
                    sleep(0.01)  # CPU 점유율 줄이기
                    continue

            if not self.post_place:
                self.running = False
                rospy.loginfo("모든 QR 코드 인식 완료. 기능 종료...")
                return

            #  프레임 스킵 적용 (5프레임마다 한 번씩 실행)
            if frame_counter % 5 != 0:  # 5프레임에 한 번씩 인식
                frame_counter += 1
                continue
            frame_counter += 1

            #  빛 반사 및 그림자 보정 적용
            processed_frame = self.preprocess_frame(frame)

            qr_codes = pyzbar.decode(processed_frame)

            cv2.imshow("qr", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break 

            if not qr_codes:  # QR 코드가 감지되지 않았을 경우
                sleep(0.01)  # CPU 사용량 줄이기
                continue

            for qr in qr_codes:
                # QR 코드 데이터 추출 (문자열 변환)
                qr_data = qr.data.decode("utf-8").strip()
                
                # QR 코드가 저장된 데이터 목록에 있고, "home"이 아닌 경우
                if qr_data in self.post_place and qr_data != "home":
                    self.post_place.remove(qr_data)  # 감지된 QR 코드 삭제
                    rospy.loginfo(f"QR 감지: {qr_data}, 남은 QR 코드 : {self.post_place}")

                    # [배송 중] 상태를 웹으로 전송
                    self.delivery_status_pub.publish(f"배송 중 (최근 배송 : {qr_data}동)")

                    # 정지 및 부저 울림
                    self.line_tracing_active = False
                    sleep(2)

                    # 부저 3회 울리기
                    for _ in range(3):
                        bus.write_i2c_block_data(ADDRESS, 0x06, [1])  # Buzzer ON
                        sleep(1)
                        bus.write_i2c_block_data(ADDRESS, 0x06, [0])  # Buzzer OFF
                        sleep(1)

                    sleep(5)

                    # 다시 출발
                    self.line_tracing_active = True

                # QR 데이터가 "home"일 경우 도착 처리
                elif qr_data == "home":
                    # 정지
                    self.line_tracing_active = False

                    # 배송지 정보 초기화
                    self.post_place = []

                    # [배송 완료] 상태를 웹으로 전송
                    self.delivery_status_pub.publish("모든 택배 배송 완료")

                    rospy.loginfo("모든 QR 코드 인식 완료. 기능 종료...")
                    rospy.loginfo(f"남은 배송지 정보 : {self.post_place}")
            
    def robotMove(self, msg):
        # 공백 제거
        command = msg.data.strip()

        if command == "go":
            # [배송 중] 상태를 웹으로 전송
            self.delivery_status_pub.publish("배송 중")
            self.running = True

            # 라인트레이싱 및 QR 코드 인식을 멀티쓰레드로 실행
            lineTracing_thread = threading.Thread(target=self.lineTracing)
            qrDetection_thread = threading.Thread(target=self.detectedQRcode)
            
            lineTracing_thread.start()
            qrDetection_thread.start()

            # 영상 송출은 하나만 유지
            while self.running and not rospy.is_shutdown():
                frame = self.camera.read()

                if frame is not None:
                    with self.lock:
                        self.shared_frame = frame.copy()
            
            # 멈추면 쓰레드 종료
            self.running = False

            # 안정적인 종료 보장
            if lineTracing_thread.is_alive():
                lineTracing_thread.join(timeout=1)
            if qrDetection_thread.is_alive():
                qrDetection_thread.join(timeout=1)
            
            cv2.destroyAllWindows()

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = PostRobot()
    node.run()