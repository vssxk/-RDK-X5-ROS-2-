import os
import xml.etree.ElementTree as ET
from glob import glob

def convert_xml_to_txt(xml_file, output_dir, class_names):
    """
    将单个XML文件转换为YOLOv5格式的TXT文件
    
    参数:
        xml_file: XML文件路径
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 获取图像尺寸
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    # 创建输出文件路径
    base_name = os.path.splitext(os.path.basename(xml_file))[0]
    txt_file = os.path.join(output_dir, base_name + '.txt')
    
    with open(txt_file, 'w') as f:
        for obj in root.iter('object'):
            # 获取类别
            cls = obj.find('name').text
            if cls not in class_names:
                continue  # 跳过不在class_names中的类别
            cls_id = class_names.index(cls)
            
            # 获取边界框坐标
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)
            
            # 转换为YOLOv5格式 (中心x, 中心y, 宽度, 高度) 归一化到[0,1]
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # 写入文件
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def batch_convert_xml_to_txt(xml_dir, output_dir, class_names):
    """
    批量转换XML文件为YOLOv5格式的TXT文件
    
    参数:
        xml_dir: 包含XML文件的目录
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有XML文件
    xml_files = glob(os.path.join(xml_dir, '*.xml'))
    
    # 转换每个文件
    for xml_file in xml_files:
        convert_xml_to_txt(xml_file, output_dir, class_names)
    
    print(f"转换完成! 共转换了 {len(xml_files)} 个文件到 {output_dir}")

if __name__ == '__main__':
    import argparse
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='将XML标注文件转换为YOLOv5格式的TXT文件')
    parser.add_argument('--xml-dir', type=str, required=True, help='包含XML文件的目录路径')
    parser.add_argument('--output-dir', type=str, required=True, help='输出TXT文件的目录路径')
    parser.add_argument('--classes', type=str, required=True, 
                       help='类别名称列表，用逗号分隔，例如 "cat,dog,person"')
    
    args = parser.parse_args()
    
    # 解析类别名称
    class_names = [name.strip() for name in args.classes.split(',')]
    
    # 执行批量转换
    batch_convert_xml_to_txt(args.xml_dir, args.output_dir, class_names)
