import os
import time
import cv2
import base64
import json
from kafka import KafkaProducer

VIDEO_DIR = "/Users/user/Documents/졸업논문/데이터셋"
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov"]
KAFKA_TOPIC = "video-frames"
BOOTSTRAP_SERVERS = "localhost:9092"
FRAME_DELAY = 0.03  # 초당 약 30프레임

def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def send_video_frames(video_path, producer):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ {video_path} 열기 실패")
        return

    print(f"📤 시작: {os.path.basename(video_path)}")
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        encoded = encode_frame(frame)
        producer.send(KAFKA_TOPIC, {"frame": encoded})
        frame_id += 1
        print(f"  → Frame {frame_id} 전송됨")
        time.sleep(FRAME_DELAY)

    cap.release()
    print(f"✅ 완료: {os.path.basename(video_path)}")

def main():
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    video_files = [f for f in os.listdir(VIDEO_DIR)
                   if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]

    if not video_files:
        print("❗ 동영상 파일이 없습니다.")
        return

    for filename in sorted(video_files):
        full_path = os.path.join(VIDEO_DIR, filename)
        send_video_frames(full_path, producer)

    producer.flush()
    print("🔚 모든 동영상 전송 완료.")

if __name__ == "__main__":
    main()

