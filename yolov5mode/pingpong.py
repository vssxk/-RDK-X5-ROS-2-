import os
import shutil
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from datetime import datetime
import subprocess
import sys
import time
import glob
import logging
from PIL import Image, ImageEnhance
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pingpong_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def enhance_image(image_path, output_path):
    """增强图像质量（亮度、对比度）并保存"""
    try:
        img = Image.open(image_path)
        
        # 随机增强参数
        brightness_factor = np.random.uniform(0.7, 1.3)
        contrast_factor = np.random.uniform(0.7, 1.3)
        
        # 应用增强
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
        
        # 保存增强后的图像
        img.save(output_path)
        return True
    except Exception as e:
        logger.error(f"图像增强失败 {image_path}: {str(e)}")
        return False

def visualize_annotations(image_path, txt_path, output_dir, max_samples=5):
    """可视化标注以验证数据质量"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        img = Image.open(image_path)
        width, height = img.size
        draw = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw.paste(img, (0, 0))
        draw = draw.convert('RGBA')
        
        # 解析标注
        bboxes = parse_txt_annotation(txt_path, width, height)
        
        # 绘制边界框
        for bbox in bboxes:
            x_min, y_min, w, h = bbox
            x_max = x_min + w
            y_max = y_min + h
            
            # 创建半透明红色框
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw_overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw_overlay = draw_overlay.copy()
            draw_overlay = draw_overlay.resize(img.size)
            
            for x in range(int(x_min), int(x_max)):
                for y in range(int(y_min), int(y_max)):
                    if (x < img.width and y < img.height and 
                        (x < x_min + 2 or x > x_max - 2 or 
                         y < y_min + 2 or y > y_max - 2)):
                        overlay.putpixel((x, y), (255, 0, 0, 128))
            
            draw = Image.alpha_composite(draw, overlay)
        
        # 保存可视化结果
        output_path = output_dir / f"vis_{image_path.name}"
        draw.save(output_path)
        return True
    except Exception as e:
        logger.error(f"标注可视化失败 {image_path}: {str(e)}")
        return False

def parse_txt_annotation(txt_path, img_width, img_height):
    """解析YOLO格式的TXT标注文件"""
    if not txt_path.exists():
        return []
    
    bboxes = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:  # YOLO格式: class x_center y_center width height
                continue
                
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                # 转换为[x_min, y_min, width, height]格式
                x_min = (x_center - w/2) * img_width
                y_min = (y_center - h/2) * img_height
                bbox_width = w * img_width
                bbox_height = h * img_height
                
                # 只处理乒乓球类别(假设类别为0)
                if class_id == 0:
                    bboxes.append([x_min, y_min, bbox_width, bbox_height])
                    
            except ValueError:
                continue
    
    return bboxes

def save_best_model_manually(train_dir, save_dir):
    """手动保存最佳模型，即使YOLO没有生成best.pt"""
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not train_dir.exists():
        logger.error(f"训练目录不存在: {train_dir}")
        return None
    
    weights_dir = train_dir / "weights"
    if not weights_dir.exists():
        logger.error(f"权重目录不存在: {weights_dir}")
        return None
    
    # 1. 优先尝试复制best.pt
    best_pt = weights_dir / "best.pt"
    if best_pt.exists():
        target_path = save_dir / f"best_{timestamp}.pt"
        shutil.copy(best_pt, target_path)
        logger.info(f"成功复制最佳模型到: {target_path}")
        return target_path
    
    # 2. 尝试复制last.pt
    last_pt = weights_dir / "last.pt"
    if last_pt.exists():
        try:
            model = torch.load(last_pt, map_location=torch.device('cpu'))
            weights = model['model'].state_dict()
            
            new_model = {
                'epoch': model['epoch'],
                'best_fitness': model.get('best_fitness', 0.0),
                'model': weights,
                'optimizer': model.get('optimizer', None),
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'training_results': model.get('training_results', None)
            }
            
            target_path = save_dir / f"manual_best_{timestamp}.pt"
            torch.save(new_model, target_path)
            logger.info(f"手动保存最佳模型到: {target_path}")
            return target_path
        except Exception as e:
            logger.error(f"手动保存模型失败: {str(e)}")
    
    # 3. 尝试复制其他模型文件
    pt_files = list(weights_dir.glob("*.pt"))
    if pt_files:
        pt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_pt = pt_files[0]
        try:
            target_path = save_dir / f"recovered_model_{timestamp}.pt"
            shutil.copy(latest_pt, target_path)
            logger.info(f"恢复模型文件: {target_path}")
            return target_path
        except Exception as e:
            logger.error(f"复制模型文件失败: {str(e)}")
    
    logger.error("无法在训练目录中找到任何模型文件")
    return None

def run_training_command(command):
    """运行训练命令并捕获输出"""
    logger.info(f"开始训练，执行命令: {command}")
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info("\n训练输出:")
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
        
        return_code = process.wait()
        
        if return_code != 0:
            logger.error(f"训练失败，错误代码: {return_code}")
            stderr = process.stderr.read()
            if stderr:
                logger.error("错误输出:")
                logger.error(stderr)
            return False
        
        logger.info("训练成功完成")
        return True
    except Exception as e:
        logger.error(f"训练过程中发生未知错误: {str(e)}")
        return False

def check_yolov5_installation():
    """检查YOLOv5是否安装正确"""
    logger.info("检查YOLOv5安装...")
    
    if not Path("yolov5").exists():
        logger.info("克隆YOLOv5仓库...")
        os.system("git clone https://github.com/ultralytics/yolov5.git")
    
    logger.info("安装依赖...")
    os.system("pip install -r yolov5/requirements.txt")
    
    required_files = [
        "yolov5/train.py",
        "yolov5/models/yolo.py",
        "yolov5/utils/general.py"
    ]
    
    for file in required_files:
        if not Path(file).exists():
            logger.error(f"关键文件缺失: {file}")
            return False
    
    logger.info("YOLOv5安装检查完成")
    return True

def check_gpu_availability():
    """检查GPU是否可用"""
    try:
        if torch.cuda.is_available():
            logger.info(f"GPU可用，设备: {torch.cuda.get_device_name(0)}")
            return True
        else:
            logger.warning("GPU不可用，将使用CPU训练，速度会很慢")
            return False
    except:
        logger.error("无法检查GPU状态")
        return False

def check_disk_space():
    """检查磁盘空间"""
    try:
        total, used, free = shutil.disk_usage(".")
        logger.info(f"磁盘空间 - 总共: {total//(2**30)} GB, 已用: {used//(2**30)} GB, 可用: {free//(2**30)} GB")
        return free > 10 * 1024**3
    except:
        logger.error("无法检查磁盘空间")
        return False

def analyze_bbox_sizes(bboxes_list):
    """分析边界框尺寸分布"""
    widths = []
    heights = []
    areas = []
    
    for bboxes in bboxes_list:
        for bbox in bboxes:
            w, h = bbox[2], bbox[3]
            widths.append(w)
            heights.append(h)
            areas.append(w * h)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    sns.histplot(widths, bins=20, kde=True)
    plt.title('Width Distribution')
    plt.xlabel('Width (pixels)')
    
    plt.subplot(132)
    sns.histplot(heights, bins=20, kde=True)
    plt.title('Height Distribution')
    plt.xlabel('Height (pixels)')
    
    plt.subplot(133)
    sns.histplot(areas, bins=20, kde=True)
    plt.title('Area Distribution')
    plt.xlabel('Area (pixels^2)')
    
    plt.tight_layout()
    plt.savefig('bbox_size_distribution.png')
    logger.info("边界框尺寸分布图已保存为 bbox_size_distribution.png")

def test_camera_with_model(model_path, conf_thres=0.3):
    """使用摄像头测试训练好的模型"""
    logger.info(f"使用摄像头测试模型: {model_path}")
    
    # 确保模型文件存在
    if not Path(model_path).exists():
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    # 加载模型
    try:
        # 使用YOLOv5的本地代码加载模型
        sys.path.append(str(Path("yolov5").absolute()))  # 添加yolov5目录到PATH
        from models.common import DetectMultiBackend
        from utils.general import non_max_suppression, scale_boxes
        from utils.augmentations import letterbox
        from utils.plots import Annotator, colors
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DetectMultiBackend(model_path, device=device, dnn=False, fp16=False)
        model.eval()
        logger.info(f"模型加载成功，置信度阈值: {conf_thres}")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        return
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("无法打开摄像头")
        return
    
    logger.info("摄像头已打开，按 'q' 键退出...")
    cv2.namedWindow('Pingpong Detection', cv2.WINDOW_NORMAL)
    
    # 获取模型参数
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = (640, 640)  # 推理尺寸
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("无法读取摄像头帧")
            break
        
        # 预处理图像
        im0 = frame.copy()
        im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]  # 调整大小和填充
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # 连续数组
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # 增加批次维度
        
        # 推理
        pred = model(im, augment=False, visualize=False)
        # NMS
        pred = non_max_suppression(pred, conf_thres, 0.45, None, False, max_det=1000)
        
        # 处理检测结果
        det = pred[0]  # 假设只有一批
        annotator = Annotator(im0, line_width=2, example=str(names))
        if len(det):
            # 将边界框从im大小调整到im0大小
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            
            # 绘制结果
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f"{names[c]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))
        
        # 显示帧
        im0 = annotator.result()
        cv2.imshow('Pingpong Detection', im0)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info("摄像头测试结束")

def main():
    logger.info("="*50)
    logger.info("乒乓球检测模型训练流程启动")
    
    # 系统检查
    logger.info("\n系统检查:")
    check_gpu_availability()
    if not check_disk_space():
        logger.warning("磁盘空间不足可能影响训练")
    
    # 1. 加载本地数据集
    logger.info("="*50)
    logger.info("开始加载数据集...")
    data_dir = Path("/app/pydev_demo/02_usb_camera_sample/pingpong/pingpong/data/ball/")
    
    # 调试：显示目录结构
    logger.info("\n当前目录结构:")
    files = list(data_dir.glob('*'))
    for i, f in enumerate(files[:5]):
        logger.info(f"  {f.name}")
    logger.info(f"(共{len(files)}个文件)")
    
    # 2. 获取所有图片文件
    image_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
    logger.info(f"\n找到 {len(image_files)} 张图片文件")
    
    if len(image_files) == 0:
        logger.error("没有找到任何图片文件！")
        logger.error(f"检查目录: {data_dir}")
        return

    # 3. 创建增强数据集目录
    enhanced_dir = data_dir / "enhanced"
    enhanced_dir.mkdir(exist_ok=True)
    
    # 4. 处理图片和标注
    data = []
    missing_labels = 0
    invalid_bboxes = 0
    valid_images = 0
    all_bboxes = []

    logger.info("\n开始处理标注文件...")
    for img_path in image_files:
        # 创建增强版本
        enhanced_path = enhanced_dir / img_path.name
        if not enhanced_path.exists():
            enhance_image(img_path, enhanced_path)
        
        # 获取图片尺寸
        try:
            with Image.open(enhanced_path) as img:
                width, height = img.size
        except Exception as e:
            logger.error(f"无法读取图片尺寸 {enhanced_path.name}: {str(e)}")
            continue
            
        txt_path = img_path.with_suffix('.txt')
        
        # 解析TXT标注文件
        bboxes = parse_txt_annotation(txt_path, width, height)
        all_bboxes.append(bboxes)
        
        # 检查是否有标注文件但解析不到有效标注
        if txt_path.exists() and not bboxes:
            logger.warning(f"空或无效的标注文件: {txt_path.name}")
            missing_labels += 1
            continue
            
        # 验证边界框
        valid_bboxes = []
        for bbox in bboxes:
            x_min, y_min, w, h = bbox
            if (0 <= x_min < width and 
                0 <= y_min < height and 
                3 <= w <= 300 and  # 放宽尺寸限制
                3 <= h <= 300 and
                x_min + w <= width and 
                y_min + h <= height):
                valid_bboxes.append(bbox)
            else:
                logger.warning(f"无效边界框 {txt_path.name}: {bbox}")
                invalid_bboxes += 1
        
        # 只保留有有效标注的图片
        if valid_bboxes:
            data.append({
                "image_path": str(enhanced_path),
                "width": width,
                "height": height,
                "bboxes": valid_bboxes
            })
            valid_images += 1
        elif txt_path.exists():
            logger.warning(f"{img_path.name} 没有有效边界框")

    # 分析边界框尺寸
    if all_bboxes:
        analyze_bbox_sizes(all_bboxes)
    
    # 5. 可视化部分标注
    vis_dir = data_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    logger.info("\n可视化部分标注...")
    for i, img_path in enumerate(image_files[:5]):  # 只可视化前5个
        txt_path = img_path.with_suffix('.txt')
        visualize_annotations(img_path, txt_path, vis_dir)
    logger.info(f"标注可视化已保存到: {vis_dir}")

    # 6. 诊断报告
    logger.info("\n" + "="*50)
    logger.info("数据集诊断报告:")
    logger.info(f"总图片数: {len(image_files)}")
    logger.info(f"有效图片数: {valid_images}")
    logger.info(f"缺失标注文件: {missing_labels}")
    logger.info(f"无效边界框: {invalid_bboxes}")
    
    if not data:
        logger.error("\n致命错误: 没有找到任何有效数据!")
        logger.error("可能原因:")
        logger.error("1. 标注文件(.txt)不存在或格式不正确")
        logger.error("2. 图片文件扩展名不是.jpg或.png")
        logger.error("3. 所有边界框都被过滤掉了")
        logger.error("4. 目录路径不正确")
        logger.error("\n建议检查:")
        logger.error(f"- 确认目录 {data_dir} 包含图片和.txt标注文件")
        logger.error("- 检查样例标注文件是否为有效的YOLO格式")
        logger.error("- 尝试放宽边界框过滤条件")
        
        # 打印样例TXT文件内容
        sample_txt = next(data_dir.glob("*.txt"), None)
        if sample_txt:
            logger.info("\n样例TXT文件内容:")
            with open(sample_txt, 'r') as f:
                logger.info(f.read())
        
        raise ValueError("没有有效的训练数据，请检查上述诊断信息")

    # 7. 准备YOLOv5数据集
    logger.info("\n" + "="*50)
    logger.info("准备YOLOv5数据集结构...")
    BASE_DIR = Path("pingpong_yolov5")
    BASE_DIR.mkdir(exist_ok=True)
    
    MODEL_SAVE_DIR = BASE_DIR / "saved_models"
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"模型将保存到: {MODEL_SAVE_DIR}")

    # 创建目录结构
    (BASE_DIR / "images/train").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "images/val").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "labels/train").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "labels/val").mkdir(parents=True, exist_ok=True)

    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 划分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    logger.info(f"\n数据集划分:")
    logger.info(f"训练集: {len(train_df)} 张图片")
    logger.info(f"验证集: {len(val_df)} 张图片")

    # 8. 创建YOLOv5数据配置文件
    data_config = {
        "path": str(BASE_DIR.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "pingpong"},
        "nc": 1
    }

    with open(BASE_DIR / "pingpong.yaml", 'w') as f:
        yaml.dump(data_config, f)
    logger.info("\n已创建YOLOv5配置文件: pingpong.yaml")

    # 9. 保存数据集（转换为YOLO格式）
    def save_yolo_dataset(df, img_dir, label_dir):
        for idx, row in df.iterrows():
            try:
                img_src = Path(row["image_path"])
                img_dst = img_dir / img_src.name
                shutil.copy(img_src, img_dst)
                
                label_path = label_dir / f"{img_src.stem}.txt"
                with open(label_path, 'w') as f:
                    for bbox in row["bboxes"]:
                        x_center = (bbox[0] + bbox[2]/2) / row["width"]
                        y_center = (bbox[1] + bbox[3]/2) / row["height"]
                        w = bbox[2] / row["width"]
                        h = bbox[3] / row["height"]
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                        
            except Exception as e:
                logger.error(f"保存 {row['image_path']} 失败: {e}")

    logger.info("\n保存训练集...")
    save_yolo_dataset(
        train_df, 
        BASE_DIR / "images/train", 
        BASE_DIR / "labels/train"
    )
    
    logger.info("保存验证集...")
    save_yolo_dataset(
        val_df, 
        BASE_DIR / "images/val", 
        BASE_DIR / "labels/val"
    )

    # 10. 训练模型
    logger.info("\n" + "="*50)
    logger.info("准备训练YOLOv5模型...")
    
    if not check_yolov5_installation():
        logger.error("YOLOv5安装不完整，无法继续训练")
        return
    
    # 改进的训练命令 - 使用YOLOv5s模型，增加epochs，添加早停
    train_cmd = (
        f"python yolov5/train.py "
        f"--img 640 " 
        f"--batch 8 "   # 增加批大小
        f"--epochs 50 "  # 增加训练轮次
        f"--data {BASE_DIR}/pingpong.yaml "
        f"--weights yolov5n.pt "  # 使用更大的模型
        f"--cache "
        f"--optimizer Adam "
        f"--patience 10 "  # 添加早停机制
        f"--cos-lr "
        f"--label-smoothing 0.1 "
        f"--bbox_interval 1 "
        f"--name pingpong_detection "
        f"--exist-ok "
        f"--project pingpong_training "
        f"--multi-scale "  # 多尺度训练
    )
    
    logger.info("\n训练命令:")
    logger.info(train_cmd)
    
    training_success = run_training_command(train_cmd)
    
    # 11. 手动保存最佳模型
    logger.info("\n" + "="*50)
    logger.info("尝试保存最佳模型...")
    
    train_output_dir = Path("pingpong_training/pingpong_detection")
    if not train_output_dir.exists():
        train_output_dir = Path("yolov5/runs/train/pingpong_detection")
    
    model_path = save_best_model_manually(train_output_dir, MODEL_SAVE_DIR)
    
    # 12. 使用摄像头测试模型
    if model_path:
        logger.info("\n" + "="*50)
        logger.info("准备使用摄像头测试训练好的模型...")
        
        # 逐步降低置信度阈值测试
        for conf in [0.5, 0.3, 0.2, 0.1]:
            logger.info(f"\n测试置信度阈值: {conf}")
            test_camera_with_model(model_path, conf_thres=conf)
            
            # 询问用户是否继续测试
            user_input = input("继续测试更低阈值? (y/n): ").strip().lower()
            if user_input != 'y':
                break
    
    # 训练失败处理
    if not training_success or not model_path:
        logger.error("\n" + "="*50)
        logger.error("训练可能未成功完成或未保存模型")
        logger.error("建议解决方案:")
        logger.error("1. 检查数据集质量（使用生成的visualizations目录）")
        logger.error("2. 减少批大小 (--batch 4)")
        logger.error("3. 尝试更小的模型 (yolov5n.pt)")
        logger.error("4. 增加训练轮次 (--epochs 100)")
        logger.error("5. 检查边界框尺寸分布图 (bbox_size_distribution.png)")
    
    logger.info("\n" + "="*50)
    logger.info("训练流程完成!")

if __name__ == "__main__":
    try:
        start_time = time.time()
        main()
        end_time = time.time()
        logger.info(f"总耗时: {(end_time - start_time)/60:.2f} 分钟")
    except Exception as e:
        logger.error(f"程序异常终止: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
