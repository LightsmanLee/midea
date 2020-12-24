# import argparse
import base64
import copy
# import glob
import json
import os
# import subprocess
import xmltodict
import requests
from progressbar import progressbar
# from collections import defaultdict
import cv2 as cv
# import matplotlib.pyplot as plt
# from PIL import Image


def xml_analysis(items):
    label_info = {'classes': [], 'scores': [], 'boxes': []}
    items = [items] if not isinstance(items, list) else items
    for item in items:
        label_info['classes'].append(item.get('name'))
        label_info['boxes'].append([
            int(item.get('bndbox').get(x))
            for x in ['xmin', 'ymin', 'xmax', 'ymax']
        ])
        label_info['scores'] = item.get('score')
    return label_info


def draw(labels, img, pre=False):
    if not pre:
        lenth = len(labels['classes'])
        classes = labels['classes']
        boxes = labels['boxes']
        color = [0, 0, 255]
        for i in range(0, lenth):
            if labels['classes'][i] != 'drawer':
                x0, y0, x1, y1 = boxes[i]
                foodname = classes[i]
                cv.rectangle(img, (x0, y0), (x1, y1), color, 1)
                #cv.rectangle(img,(x0,y0),(int(x0+(x1-x0)*0.7),y0+15),color,1)
                cv.putText(img, foodname, (x0 + 2, y0 + 10),
                           cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    else:
        lenth = len(labels['classes'])
        classes = labels['classes']
        boxes = labels['boxes']
        color = [0, 255, 0]
        for i in range(0, lenth):
            x0, y0, x1, y1 = boxes[i]
            foodname = classes[i]
            cv.rectangle(img, (x0, y0), (x1, y1), color, 1)
            #cv.rectangle(img,(x0,y1-15),(int(x0+(x1-x0)*0.7),y1),color,1)
            cv.putText(img, foodname, (x0 + 2, y1 - 4), cv.FONT_HERSHEY_PLAIN,
                       1, (0, 255, 0), 1)


def reverse_coordinate(item):
    lenth = len(item['classes'])
    classes = item['classes']
    boxes = item['boxes']
    for i in range(lenth):
        boxes[i] = [
            boxes[i][0] + 120, boxes[i][1] + 190, boxes[i][2] + 120,
            boxes[i][3] + 190
        ]


def files_walk(root, name='.jpg'):
    result = []
    for root, dirs, files in os.walk(root):
        for file in files:
            file_name = os.path.join(root, file)
            if name in file_name:
                result.append(file_name)
    return result


if __name__ == '__main__':
    root = './test_img'
    jpg_files = files_walk(root)
    for jpg_file in progressbar(jpg_files):
        print(jpg_file)
        xml_file = jpg_file.replace('.jpg', '.xml')
        img = cv.imread(jpg_file)
        img_crop = img[190:720, 120:1200]
        data = cv.imencode('.jpg', img_crop)[1].tostring()
        data = base64.b64encode(data)
        data = data.decode('utf-8')
        response_json = {
            'crc_result': 'fridge-1',
            'customerId': '000001',
            'source': 'ref',
            'sessionId': 'NA',
            'images': {
                '0': data
            },
        }
        headers = {
            'content-type': 'application/json',
            'Accept-Charset': 'UTF-8'
        }
        r = requests.post('http://47.99.36.241:8085/analysis',
                          data=json.dumps(response_json),
                          headers=headers)
        result = json.loads(r.text).get('data')
        folder, filename = os.path.split(jpg_file)
        ann = {
            'annotation': {
                'folder': '',
                'filename': '',
                'path': '',
                'source': {
                    'database': 'Unknown'
                },
                'size': {
                    'width': 1280,
                    'height': 720,
                    'depth': 3
                },
                'segmented': 0,
                'object': []
            }
        }
        obj = {
            'name': '',
            'pose': 'Unspecified',
            'truncated': 0,
            'difficult': '0',
            'bndbox': {
                'xmin': 0,
                'ymin': 0,
                'xmax': 0,
                'ymax': 0
            }
        }
        ann['annotation']['folder'] = folder
        ann['annotation']['filename'] = filename
        ann['annotation']['path'] = jpg_file
        result_data = eval(result).get('classifier')
        crc = True
        for item in result_data:
            if crc:
                obj['name'] = item.get('food')
            else:
                obj['name'] = item.get('category')
            obj['bndbox']['xmin'] = int(item.get('shape').get('xmin'))
            obj['bndbox']['ymin'] = int(item.get('shape').get('ymin'))
            obj['bndbox']['xmax'] = int(item.get('shape').get('xmax'))
            obj['bndbox']['ymax'] = int(item.get('shape').get('ymax'))
            obj['score'] = item.get('prod')
            ann['annotation']['object'].append(copy.deepcopy(obj))
        predict = ann['annotation']['object']
        with open(xml_file, 'r', encoding='utf-8') as file:
            xml_str = file.read()
        xml_parse = xmltodict.parse(xml_str)
        target = xml_parse.get('annotation').get('object')
        predict = xml_analysis(predict)
        reverse_coordinate(predict)
        target = xml_analysis(target)
        # img_copy=img.copy()
        draw(predict, img, True)
        draw(target, img, False)
        # img=np.hstack((img,img_copy))
        path = os.path.join(
            './test_img_draw',
            os.path.relpath(os.path.split(jpg_file)[0], './test_img'))
        if os.path.exists(path) is not True:
            os.makedirs(path)
        cv.imwrite(os.path.join(path, os.path.split(jpg_file)[1]), img)
        # cv.imshow('img',img_copy)
        # key=cv.waitKey(0)
        # if key==27:
        # cv.destroyAllWindows()
        # break
        # cv.destroyAllWindows()
