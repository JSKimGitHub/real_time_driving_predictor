from kafka import KafkaConsumer
import cv2
import numpy as np
import base64
import json

def get_frame_from_kafka(topic='video-frames', bootstrap_servers='localhost:9092'):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
	auto_offset_reset='earliest',
	enable_auto_commit=True,
	group_id='frame-group'
    )
    for message in consumer:
        frame_bytes = base64.b64decode(message.value['frame'])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        yield frame

