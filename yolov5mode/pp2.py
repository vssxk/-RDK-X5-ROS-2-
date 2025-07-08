import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import time

# 设置 yolov5 路径
yolov5_path = str(Path("yolov5").resolve())
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

# 兼容不同版本YOLOv5的导入
try:
    from utils.general import scale_coords as scale_boxes
    from utils.general import non_max_suppression
    from utils.general import letterbox
except ImportError:
    from utils.general import scale_boxes
    from utils.general import non_max_suppression
    from utils.augmentations import letterbox

from models.experimental import attempt_load
from utils.torch_utils import select_device

# 初始化参数
weights = 'yolov5n.pt'  # 模型权重文件路径
device = select_device('')  # 自动选择设备 (CPU或CUDA)
conf_thres = 0.25  # 置信度阈值
iou_thres = 0.45  # IOU阈值
imgsz = 640  # 输入图像大小

# 加载模型
model = attempt_load(weights, device=device)
stride = int(model.stride.max())  # 模型步长
names = model.module.names if hasattr(model, 'module') else model.names  # 获取类别名称

# 尝试不同的摄像头后端
def open_camera(index):
    # 尝试使用不同的后端
    for api in [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            print(f"使用后端 {api} 成功打开摄像头")
            return cap
    return None

# 打开摄像头
cap = open_camera(0)

# 检查摄像头是否成功打开
if cap is None or not cap.isOpened():
    # 尝试其他摄像头索引
    print("无法打开默认摄像头，尝试其他索引...")
    for i in range(1, 4):
        cap = open_camera(i)
        if cap and cap.isOpened():
            print(f"成功打开摄像头索引 {i}")
            break
    
    if cap is None or not cap.isOpened():
        raise IOError("无法打开任何摄像头")

# 不设置分辨率，让OpenCV自动选择摄像头支持的分辨率
# 获取实际分辨率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"摄像头分辨率: {width}x{height}, FPS: {fps:.2f}")

# 创建显示窗口
cv2.namedWindow('YOLOv5 物体检测', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLOv5 物体检测', 800, 600)

# 最大重试次数
max_retries = 5
retry_delay = 0.1  # 秒

# 处理每一帧
frame_count = 0
start_time = time.time()

while True:
    # 读取帧，带有重试机制
    ret = False
    frame = None
    for _ in range(max_retries):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            break
        time.sleep(retry_delay)
    
    if not ret or frame is None or frame.size == 0:
        print("无法获取有效帧，跳过...")
        # 尝试重新初始化摄像头
        cap.release()
        time.sleep(1)
        cap = open_camera(0)
        if cap is None or not cap.isOpened():
            print("无法重新初始化摄像头，退出...")
            break
        continue
    
    # 原始图像尺寸
    h, w = frame.shape[:2]
    
    # 每10帧显示一次处理速度
    frame_count += 1
    if frame_count % 10 == 0:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        print(f"处理速度: {fps:.2f} FPS")
        frame_count = 0
        start_time = time.time()
    
    # 预处理图像
    img = letterbox(frame, imgsz, stride=stride)[0]  # 调整大小并填充
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR转RGB，HWC转CHW
    img = np.ascontiguousarray(img)  # 转换为连续数组
    
    # 转换为torch张量并送到设备
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # 归一化 (0 - 255 to 0.0 - 1.0)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # 添加批次维度
    
    # 推理
    pred = model(img, augment=False)[0]
    
    # NMS (非极大值抑制)
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    # 处理检测结果
    for det in pred:  # 每张图像的检测结果
        if len(det):
            # 将边界框从img_size缩放到原始图像大小
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            
            # 绘制结果
            for *xyxy, conf, cls in reversed(det):
                # 边界框坐标
                x1, y1, x2, y2 = map(int, xyxy)
                
                # 绘制边界框
                color = (0, 255, 0)  # 绿色
                thickness = 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # 标签文本
                label = f'{names[int(cls)]} {conf:.2f}'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # 标签背景
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), 
                             (x1 + label_size[0], y1), color, -1)
                
                # 标签文本
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 显示结果
    cv2.imshow('YOLOv5 物体检测', frame)
    
    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
