import glob
import json
import os
import numpy as np
from progressbar import progressbar
import xml.etree.ElementTree as ET
import cv2 as cv
from collections import defaultdict

'''
parse_rec:品类是大类还是小类
'''
def fps_fns_img(detfile,
                imagenames,
                classname,
                ovthresh=0.5,
                use_07_metric=False):
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(imagename)
    class_recs = {}  # 保存的是 Ground Truth的数据
    npos = 0
    for imagename in imagenames:
        # 获取Ground Truth每个文件中某种类别的物体
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        #  different基本都为0/False
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)  # 自增，~difficult取反,统计样本个数
        # # 记录Ground Truth的内容
        class_recs[imagename] = {
            'bbox': bbox,
            'difficult': difficult,
            'det': det
        }
    image_ids = [x[0] for x in detfile]  # 图片ID
    confidence = np.array([float(x[1]) for x in detfile])
    BB = np.array([[float(z) for z in x[2:]]
                   for x in detfile])  # bounding box数值

    # 对confidence的index根据值大小进行降序排列。
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    # 重排bbox，由大概率到小概率。
    BB = BB[sorted_ind, :]
    # 图片重排，由大概率到小概率。
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    fps_img = []
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
                    fps_img.append(image_ids[d])
        else:
            fp[d] = 1.
            fps_img.append(image_ids[d])
    fns_img = []
    for key in class_recs:
        if sum(class_recs[key]['det']) == len(class_recs[key]['det']):
            continue
        else:
            fns_img.append(key)
    return fps_img, fns_img


def fps_fns_class(detfile,
                  imagenames,
                  classname,
                  ovthresh=0.5,
                  use_07_metric=False):
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(imagename)
    class_recs = {}  # 保存的是 Ground Truth的数据
    npos = 0
    for imagename in imagenames:
        # 获取Ground Truth每个文件中某种类别的物体
        R = [obj for obj in recs[imagename]]
        bbox = np.array([x['bbox'] for x in R])
        names = np.array([x['name'] for x in R])
        #  different基本都为0/False
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)  # 自增，~difficult取反,统计样本个数
        # # 记录Ground Truth的内容
        class_recs[imagename] = {
            'bbox': bbox,
            'difficult': difficult,
            'det': det,
            'name': names
        }
    image_ids = [x[0] for x in detfile]  # 图片ID
    confidence = np.array([float(x[1]) for x in detfile])
    BB = np.array([[float(z) for z in x[2:]]
                   for x in detfile])  # bounding box数值

    # 对confidence的index根据值大小进行降序排列。
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    # 重排bbox，由大概率到小概率。
    BB = BB[sorted_ind, :]
    # 图片重排，由大概率到小概率。
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    if R['name'][jmax] == classname:
                        R['det'][jmax] = 1
                    else:
                        fps_class[classname].add(R['name'][jmax])
                        fns_class[R['name'][jmax]].add(classname)
                else:
                    fps_class[classname].add(None)
        else:
            fps_class[classname].add(None)
    for key in class_recs:
        for x in range(len(class_recs[key]['det'])):
            if class_recs[key]['det'][x] == 0:
                fns_class[class_recs[key]['name'][x]].add(None)
    return


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    # 解析xml文件，将GT框信息放入一个列表
    for obj in tree.findall('object'):
        if obj.find('name') != 'drawer':
            obj_struct = {}
            # obj_struct['name'] = obj.find('name').text.split('_')[0].replace(
            #     '-', '_')
            obj_struct['name'] = obj.find('name').text
            if obj_struct['name'] == 'little_taro':
                obj_struct['name'] = 'taro'
            elif obj_struct['name'] == 'mango_sharp_mouth':
                obj_struct['name'] = 'mango'
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [
                int(bbox.find('xmin').text),
                int(bbox.find('ymin').text),
                int(bbox.find('xmax').text),
                int(bbox.find('ymax').text)
            ]
            objects.append(obj_struct)
    return objects


def files_walk(root, name='.jpg'):
    result = []
    for root, dirs, files in os.walk(root):
        for file in files:
            file_name = os.path.join(root, file)
            if name in file_name:
                result.append(file_name)
    return result


def get_predict_classes(txt_files):
    predict_classes = {}
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            file = json.load(f)
        predict_classes[os.path.basename(txt_file).split('.')[0]] = file
    return predict_classes


if __name__ == '__main__':
    txt_files = glob.glob(os.path.join('./predict_result', '*.txt'))
    predict_classes = get_predict_classes(txt_files)

    with open('./foodname/target_name.txt', 'r') as f:
        badfood = f.read()
    badfood = badfood.split('\n')
    # badfood = ['apple']

    root = r'./test_img'
    xml_files = files_walk(root, '.xml')
    fps_class = defaultdict(set)
    fns_class = defaultdict(set)

    for foodname in progressbar(badfood):
        fps, fns = fps_fns_img(predict_classes[foodname], xml_files, foodname)
        for fp in fps:
            fp = fp.replace('.xml', '.jpg')
            path = os.path.join('./test_img_draw',
                                os.path.relpath(fp, './test_img'))
            img = cv.imread(path)
            newpath = os.path.join('./faultpic', foodname, 'fp',
                                   os.path.basename(path))
            if not os.path.exists(os.path.join('./faultpic', foodname, 'fp')):
                os.makedirs(os.path.join('./faultpic', foodname, 'fp'))
            cv.imwrite(newpath, img)
        for fn in fns:
            fn = fn.replace('.xml', '.jpg')
            path = os.path.join('./test_img_draw',
                                os.path.relpath(fn, './test_img'))
            img = cv.imread(path)
            newpath = os.path.join('./faultpic', foodname, 'fn',
                                   os.path.basename(path))
            if not os.path.exists(os.path.join('./faultpic', foodname, 'fn')):
                os.makedirs(os.path.join('./faultpic', foodname, 'fn'))
            cv.imwrite(newpath, img)
        fps_fns_class(predict_classes[foodname], xml_files, foodname)
