import glob
import json
import os
import xmltodict
from progressbar import progressbar
import cv2 as cv
from collections import defaultdict


def transform(classname, class_txt, predict_dict):
    for i in range(len(class_txt)):
        if 'classes' not in predict_dict[class_txt[i][0]]:
            predict_dict[class_txt[i][0]]['classes'] = []
        if 'scores' not in predict_dict[class_txt[i][0]]:
            predict_dict[class_txt[i][0]]['scores'] = []
        if 'boxes' not in predict_dict[class_txt[i][0]]:
            predict_dict[class_txt[i][0]]['boxes'] = []
        predict_dict[class_txt[i][0]]['classes'].append(classname)
        predict_dict[class_txt[i][0]]['scores'].append(class_txt[i][1])
        predict_dict[class_txt[i][0]]['boxes'].append(class_txt[i][2:])


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
                # cv.rectangle(img,(x0,y0),(int(x0+(x1-x0)*0.7),y0+15),color,1)
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
            # cv.rectangle(img,(x0,y1-15),(int(x0+(x1-x0)*0.7),y1),color,1)
            cv.putText(img, foodname, (x0 + 2, y1 - 4), cv.FONT_HERSHEY_PLAIN,
                       1, (0, 255, 0), 1)


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


def files_walk(root):
    result = []
    for root, dirs, files in os.walk(root):
        for file in files:
            file_name = os.path.join(root, file)
            if '.jpg' in file_name:
                result.append(file_name)
    return result


if __name__ == '__main__':
    # 得到预测字典
    path = './predict_result'
    txt_files = glob.glob(os.path.join(path, '*.txt'), recursive=True)
    predict_dict = defaultdict(defaultdict)
    for txt_file in txt_files:
        foodname = os.path.split(txt_file)[1].split('.')[0]
        with open(txt_file, 'r') as f:
            file = json.load(f)
        transform(foodname, file, predict_dict)

    # 绘制框图
    root = './test_img'
    jpg_files = files_walk(root)

    for jpg_file in progressbar(jpg_files):
        xml_file = jpg_file.replace('.jpg', '.xml')
        img = cv.imread(jpg_file)
        with open(xml_file, 'r', encoding='utf-8') as file:
            xml_str = file.read()
        xml_parse = xmltodict.parse(xml_str)
        target = xml_parse.get('annotation').get('object')
        target = xml_analysis(target)
        predict = predict_dict[xml_file]
        if len(predict) > 0:
            draw(predict, img, True)
        if len(target) > 0:
            draw(target, img, False)
        # img=np.hstack((img,img_copy))
        path = os.path.join(
            './test_img_draw',
            os.path.relpath(os.path.split(jpg_file)[0], './test_img'))
        if os.path.exists(path) is not True:
            os.makedirs(path)
        cv.imwrite(os.path.join(path, os.path.split(jpg_file)[1]), img)