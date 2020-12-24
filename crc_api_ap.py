# from evaluate_ap import voc_ap
from evaluate_ap import voc_eval
# from evaluate_ap import parse_rec
# import argparse
import base64
# import copy
# import glob
import json
import os
# import subprocess
# import xmltodict
import requests
# import xmltodict
# import numpy as np
from progressbar import progressbar
import xml.etree.ElementTree as ET
import cv2 as cv
# import matplotlib.pyplot as plt
# from PIL import Image
# import time
from collections import defaultdict
import pandas as pd
# from collections import Counter
# import matplotlib.pyplot as plt
# import copy
# import shutil
# from collections import OrderedDict

# def parse_rec(filename):
#     """ Parse a PASCAL VOC xml file """
#     tree = ET.parse(filename)
#     objects = []
#     # 解析xml文件，将GT框信息放入一个列表
#     for obj in tree.findall('object'):
#         if obj.find('name') != 'drawer':
#             obj_struct = {}
#             obj_struct['name'] = obj.find('name').text.split('_')[0].replace(
#                 '-', '_')
#             if obj_struct['name'] == 'little_taro':
#                 obj_struct['name'] = 'taro'
#             elif obj_struct['name'] == 'mango_sharp_mouth':
#                 obj_struct['name'] = 'mango'
#             obj_struct['pose'] = obj.find('pose').text
#             obj_struct['truncated'] = int(obj.find('truncated').text)
#             obj_struct['difficult'] = int(obj.find('difficult').text)
#             bbox = obj.find('bndbox')
#             obj_struct['bbox'] = [
#                 int(bbox.find('xmin').text),
#                 int(bbox.find('ymin').text),
#                 int(bbox.find('xmax').text),
#                 int(bbox.find('ymax').text)
#             ]
#             objects.append(obj_struct)
#     return objects


def files_walk(root):
    result = []
    for root, dirs, files in os.walk(root):
        for file in files:
            file_name = os.path.join(root, file)
            if '.jpg' in file_name:
                result.append(file_name)
    return result


def request_crc(imgid, data, classes_pred):
    response_json = {
        'crc_result': 'fridge-1',
        'customerId': '000001',
        'source': 'ref',
        'sessionId': 'NA',
        'images': {
            '0': data
        },
    }
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post('http://47.99.36.241:8085/analysis',
                      data=json.dumps(response_json),
                      headers=headers)
    result = json.loads(r.text).get('data')
    result_data = eval(result).get('classifier')
    for item in result_data:
        temp = [imgid]
        temp.append(item.get('prod'))
        temp.append(int(item.get('shape').get('xmin')) + 120)
        temp.append(int(item.get('shape').get('ymin')) + 115)
        temp.append(int(item.get('shape').get('xmax')) + 120)
        temp.append(int(item.get('shape').get('ymax')) + 115)
        classes_pred[item.get('food')].append(temp)
    return


def correct_coordinate(pred):
    for key in pred:
        for i in range(len(pred[key])):
            pred[key][i][3] += 75
            pred[key][i][5] += 75
    return pred


if __name__ == '__main__':
    classes_pred = defaultdict(list)
    root = './test_img'
    jpg_files = files_walk(root)
    xml_files = [x.replace('.jpg', '.xml') for x in jpg_files]

    for jpg_file in progressbar(jpg_files):
        xml_file = jpg_file.replace('.jpg', '.xml')
        img = cv.imread(jpg_file)
        # data = cv.imencode('.jpg', img)[1].tostring()
        img_crop = img[190:670, 120:1200]  #[115:650,120:1200]
        data = cv.imencode('.jpg', img_crop)[1].tostring()
        data = base64.b64encode(data)
        data = data.decode('utf-8')
        request_crc(xml_file, data, classes_pred)

    print('predict', classes_pred)
    # 如果使用裁剪，则需要恢复坐标
    correct_coordinate(classes_pred)

    for key in classes_pred:
        if not os.path.exists('./predict_result'):
            os.makedirs('./predict_result')
        file = os.path.join('./predict_result', key + '.txt')
        with open(file, 'w') as f:
            json.dump(classes_pred[key], f)

    with open('./foodname/target_name.txt', 'r') as f:
        foodnames = f.read()
    foodnames = foodnames.split('\n')

    map = 0
    final_ap_result = []
    for name in progressbar(foodnames):
        try:
            temp = voc_eval(classes_pred[name],
                            xml_files,
                            name,
                            ovthresh=0.5,
                            use_07_metric=False)
            rec, prec, f1, ap = temp
            print(name, rec[-1], prec[-1], f1[-1], ap)
            final_ap_result.append([name, rec[-1], prec[-1], f1[-1], ap])
        except:
            print(name, 0, 0, 0, 0)
    map += ap
    final_ap_df = pd.DataFrame(
        final_ap_result,
        columns=['foodnames', 'recall', 'precision', 'f1score', 'ap'])
    final_ap_df.to_excel('calculate_ap.xlsx', index=False)
