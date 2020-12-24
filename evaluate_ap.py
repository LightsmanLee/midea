# import copy
import numpy as np
import xml.etree.ElementTree as ET
# import matplotlib.pyplot as plt


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    # 解析xml文件，将GT框信息放入一个列表
    for obj in tree.findall('object'):
        if obj.find('name') != 'drawer':
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text.split('_')[0].replace(
                '-', '_')
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


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    # 如果使用2017年的计算AP的方式(插值的方式)
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # 使用2010年后的计算AP值的方式
        # 这里是新增一个(0,0)，方便计算
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detfile,
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
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    f1 = 2 * rec * prec / (rec + prec)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, f1, ap
