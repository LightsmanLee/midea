from evaluate_ap import voc_eval
import base64
import glob
import json
import os
import requests
from progressbar import progressbar
import xml.etree.ElementTree as ET
import cv2 as cv
from collections import defaultdict
import pandas as pd
'''
find_gts:名字是否是大类还是小类；
evaluate_ap:parse_rec名字是否是大类还是小类

'''


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
        temp.append(int(item.get('shape').get('xmin')))
        temp.append(int(item.get('shape').get('ymin')))
        temp.append(int(item.get('shape').get('xmax')))
        temp.append(int(item.get('shape').get('ymax')))
        classes_pred[item.get('food')].append(temp)
    return


def correct_coordinate(pred):
    for key in pred:
        for i in range(len(pred[key])):
            pred[key][i][3] += 120
            pred[key][i][5] += 190
    return pred


def find_gts(filename, gts_counts_dict):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    # 解析xml文件，将GT框信息放入一个列表
    for obj in tree.findall('object'):
        if obj.find('name') != 'drawer':
            # foodname = obj.find('name').text.split('_')[0].replace('-', '_')
            foodname = obj.find('name').text  #标记的为35类果蔬，没有子类
            if foodname == 'little_taro':
                foodname = 'taro'
            elif foodname == 'mango_sharp_mouth':
                foodname = 'mango'
            gts_counts_dict[foodname] += 1
    return


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


if __name__ == '__main__':
    classes_pred = defaultdict(list)
    root = './test_img'
    jpg_files = files_walk(root)
    xml_files = [x.replace('.jpg', '.xml') for x in jpg_files]

    gts_counts_dict = defaultdict(int)
    for xml_file in xml_files:
        find_gts(xml_file, gts_counts_dict)

    if os.path.exists('./predict_result'):
        path = './predict_result'
        txt_files = glob.glob(os.path.join(path, '*.txt'), recursive=True)
        classes_pred = defaultdict(list)
        for txt_file in txt_files:
            foodname = os.path.split(txt_file)[1].split('.')[0]
            with open(txt_file, 'r') as f:
                file = json.load(f)
            classes_pred[foodname] = file
    else:
        for jpg_file in progressbar(jpg_files):
            xml_file = jpg_file.replace('.jpg', '.xml')
            img = cv.imread(jpg_file)
            data = cv.imencode('.jpg', img)[1].tostring()
            # img_crop = img[190:670, 120:1200]  #[115:650,120:1200]
            # data = cv.imencode('.jpg', img_crop)[1].tostring()
            data = base64.b64encode(data)
            data = data.decode('utf-8')
            request_crc(xml_file, data, classes_pred)

    # print('predict', classes_pred)
    # 如果使用裁剪，则需要恢复坐标
    # correct_coordinate(classes_pred)

    dets_counts_dict = defaultdict(int)
    for foodname in classes_pred:
        dets_counts_dict[foodname] = len(classes_pred[foodname])

    for key in classes_pred:
        if not os.path.exists('./predict_result'):
            os.makedirs('./predict_result')
        file = os.path.join('./predict_result', key + '.txt')
        with open(file, 'w') as f:
            json.dump(classes_pred[key], f)

    with open('./target_name.txt', 'r') as f:
        foodnames = f.read()
    foodnames = foodnames.split('\n')

    map = 0
    final_ap_result = []
    for name in progressbar(foodnames):
        print('\n')
        try:
            temp = voc_eval(classes_pred[name],
                            xml_files,
                            name,
                            ovthresh=0.5,
                            use_07_metric=False)
            rec, prec, f1, ap = temp
            print(name, rec[-1], prec[-1], f1[-1], ap, '\n')
            final_ap_result.append([
                name, gts_counts_dict[name], dets_counts_dict[name], rec[-1],
                prec[-1], f1[-1], ap
            ])
        except:
            print(name, 0, 0, 0, 0)
    map += ap
    final_ap_df = pd.DataFrame(final_ap_result,
                               columns=[
                                   'foodname', 'gts', 'dets', 'recall',
                                   'precision', 'f1score', 'ap'
                               ])
    final_ap_df = final_ap_df.sort_values(by='foodname')
    final_ap_df = final_ap_df.fillna(0)
    final_ap_df.to_excel('calculate_ap.xlsx', index=False)
