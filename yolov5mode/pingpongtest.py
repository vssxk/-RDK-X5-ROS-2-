import cv2
import torch
import numpy as np
from pathlib import Path
import sys

# 设置 yolov5 路径
yolov5_path = str(Path("yolov5").resolve())
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

# 兼容不同版本YOLOv5的导入
try:
    from utils.general import scale_coords as scale_boxes  # 旧版
    from utils.general import non_max_suppression
    from utils.general import letterbox  # 旧版位置
except ImportError:
    from utils.general import scale_boxes  # 新版
    from utils.general import non_max_suppression
    from utils.augmentations import letterbox  # 新版位置

from models.experimental import attempt_load
from utils.torch_utils import select_device

# 初始化设备
device = select_device("")
half = device.type != "cpu"

# 加载模型
model_path = "pingpong_yolov5/saved_models/best_20250704_212051.pt"
model = attempt_load(model_path, device=device)
if half:
    model.half()

# 类别标签
names = ["pingpong"]

# 摄像头设置
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

# 检测参数
conf_thres = 0.1
iou_thres = 0.45
img_size = 640

def detect_ball(frame):
    """处理单帧检测"""
    # 图像预处理
    img = letterbox(frame, img_size, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    # 转为张量
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 推理
    with torch.no_grad():
        pred = model(img)[0]

    # NMS处理
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # 解析结果
    detections = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f"{names[int(cls)]} {conf:.2f}"
                detections.append((xyxy, label))

    return detections

# 主循环
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_ball(frame)

    # 绘制结果
    for (xyxy, label) in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Pingpong Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
