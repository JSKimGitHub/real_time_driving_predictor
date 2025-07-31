import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO
from models.fm_ftrl_ewc import FM_FTRL_WithClassifier, EWCWrapper
from kafka_utils.consumer import get_frame_from_kafka
from kafka_utils.producer import send_prediction_to_kafka
import numpy as np
import cv2

def preprocess_input(cls_id, conf, dist, area, cx, cy, num_classes=10):
    x = np.zeros(num_classes + 5)
    if 0 <= cls_id < num_classes:
        x[cls_id] = 1.0
    x[num_classes:] = [conf, dist / 100, area / 10000, cx / 1920, cy / 1080]
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

def add_noise(x, level=0.1):
    return x + level * torch.randn_like(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("yolov8n.pt")
input_dim = 10 + 5
fm_model = FM_FTRL_WithClassifier(input_dim)
optimizer = optim.Adam(fm_model.parameters(), lr=1e-3)
ewc = EWCWrapper(fm_model)

fm_model.to(device)
fm_model.train()
loss_fn = nn.MSELoss()

frame_iter = get_frame_from_kafka()

for i, frame in enumerate(frame_iter):
    result = model(frame)[0]
    h, w, _ = frame.shape
    frame_actions = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        est_dist = 700 * 1.7 / (y2 - y1 + 1e-6)

        x = preprocess_input(cls_id, conf, est_dist, area, cx, cy).to(device)
        x_noisy = add_noise(x)

        y_pred, emb_orig = fm_model(x)
        _, emb_noisy = fm_model(x_noisy)

        loss = loss_fn(emb_noisy, emb_orig.detach()) + ewc.penalty()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        act = int(y_pred.item() >= 0.5)
        frame_actions.append(act)

    # 다수결 예측
    final_act = 1 if frame_actions and sum(frame_actions)/len(frame_actions) >= 0.5 else 0
    send_prediction_to_kafka(final_act)

    if i % 100 == 0:
        print(f"[{i}th frame] 행동: {'주행' if final_act else '멈춤'}")

