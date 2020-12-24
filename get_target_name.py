import os
import xml.etree.ElementTree as ET
from collections import defaultdict


def files_walk(root, name='.jpg'):
    result = []
    for root, dirs, files in os.walk(root):
        for file in files:
            file_name = os.path.join(root, file)
            if name in file_name:
                result.append(file_name)
    return result


def get_target_name(filename, frame_count):
    tree = ET.parse(filename)
    object_name = set()
    # 解析xml文件，将GT框信息放入一个列表
    for obj in tree.findall('object'):
        if obj.find('name').text != 'drawer':
            name_ = obj.find('name').text.split('_')[0].replace('-', '_')
            if name_ == 'little_taro':
                name_ = 'taro'
            elif name_ == 'mango_sharp_mouth':
                name_ = 'mango'
            frame_count[name_] += 1
            object_name.add(name_)
    return object_name


if __name__ == '__main__':
    root = r'E:\Imagedata_unzip'
    xml_files = files_walk(root, '.xml')

    target_name = set()
    frame_count = defaultdict(int)
    for xml_file in xml_files:
        target_name = target_name | get_target_name(xml_file, frame_count)
    target_name = list(target_name)
    target_name = [str(item) for item in target_name]

    if not os.path.exists('./foodname'):
        os.makedirs('./foodname')
    with open('./foodname/target_name.txt', 'w') as f:
        f.write('\n'.join(target_name))
