# -*- coding: utf-8 -*
import caffe
import lmdb
import numpy as np
import cv2
from caffe.proto import caffe_pb2


import numpy as np
import matplotlib

matplotlib.use('Agg')
import os
import xml.dom.minidom
# %matplotlib inline

import sys
import cv2

sys.path.insert(0, 'python')

import caffe
import shutil

# def iou(gxmin, gxmax, gymin, gymax, pxmin, pxmax, pymin, pymax):
def IOU(Reframe, GTframe, norm):
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2] - Reframe[0]
    height1 = Reframe[3] - Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2] - GTframe[0]
    height2 = GTframe[3] - GTframe[1]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)

    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if norm:
        width +=1
        height +=1

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = (width) * (height)
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)

        #print "norm", norm, ":",width, height, Area, (Area1 + Area2 - Area), ratio

    # return IOU
    return ratio


def read_xml(path):
    xmin, xmax, ymin, ymax = 0, 0, 0, 0
    with open(path, 'r') as f:
        for s in f.readlines():
            if 'xmin' in s:
                s = s.split(' ')[-1]
                # print(s[6:-8])
                xmin = int(s[6:-8])
            elif 'xmax' in s:
                s = s.split(' ')[-1]
                # print(s[6:-8])
                xmax = int(s[6:-8])
            elif 'ymin' in s:
                s = s.split(' ')[-1]
                # print(s[6:-8])
                ymin = int(s[6:-8])
            elif 'ymax' in s:
                s = s.split(' ')[-1]
                # print(s[6:-8])
                ymax = int(s[6:-8])
    return [xmin, ymin, xmax, ymax]


caffe.set_device(0)
caffe.set_mode_gpu()

model_weights = "/home/aiserver/code/source/caffe/models/ResNet/WORD/pipline_data_v900_390_1_3_re_256x256/ResNetBody_rev900_390_1_3_re_pipline_data_v900_390_1_3_re_256x256_iter_120000.caffemodel"
model_def = "/home/aiserver/code/source/caffe/models/ResNet/WORD/pipline_data_v900_390_1_3_re_256x256/deploy.prototxt"


scales = [(256, 256)]

net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
print(net.blobs['data'].data.shape)

idx = 1
# if os.path.exists('/home/aiserver/result/word_dect/res_img/'):
#     shutil.rmtree('/home/aiserver/result/word_dect/res_img/')
#
# os.mkdir('/home/aiserver/result/word_dect/res_img/')

lmdb_env = lmdb.open('/media/aiserver/disk2/word_dect/pipline_data_v2/lmdb/pipline_data_v2_test_lmdb')

lmdb_txn = lmdb_env.begin()                                 # 生成处理句柄
lmdb_cursor = lmdb_txn.cursor()                             # 生成迭代器指针
annotated_datum = caffe_pb2.AnnotatedDatum()                # AnnotatedDatum结构
#
# bei_x = 100000
# bei_y = 100000
# for bei_x in range()

bei_x = 0
bei_y = 0
idx = 0
for off_set in [2, 3, 4]:
    for bei_xx in range(0,1):
        for bei_yy in [6, 7, 8, 9]:
            res = open("---new_ptlogResNet256-lr0.001-100000-0.5-deconv-inputw900_390_re.txt", "a")
            analyze_scores = []
            analyze_iou = []

            bei_y = bei_yy * 1.0  / 20.0
            bei_x = bei_xx * 1.0 / 20.0

            idx = 0
            for key, value in lmdb_cursor:
                idx += 1

                print bei_x, bei_y, key



                # if idx ==10:
                #     break
                annotated_datum.ParseFromString(value)
                datum = annotated_datum.datum                           # Datum结构
                grps = annotated_datum.annotation_group                 # AnnotationGroup结构
                type = annotated_datum.type

                for grp in grps:
                    gt_xmin_f = grp.annotation[0].bbox.xmin
                    gt_ymin_f = grp.annotation[0].bbox.ymin
                    gt_xmax_f = grp.annotation[0].bbox.xmax
                    gt_ymax_f = grp.annotation[0].bbox.ymax

                    gt_xmin = (grp.annotation[0].bbox.xmin * datum.width)           # Annotation结构
                    gt_ymin = (grp.annotation[0].bbox.ymin * datum.height)
                    gt_xmax = (grp.annotation[0].bbox.xmax * datum.width)
                    gt_ymax = (grp.annotation[0].bbox.ymax * datum.height)

                    # print "label:", grp.group_label                            # object的name标签
                    # print "bbox:", gt_xmin, gt_ymin, gt_xmax, gt_ymax                      # object的bbox标签

                label = datum.label                                      # Datum结构label以及三个维度
                channels = datum.channels
                height = datum.height
                width = datum.width
                #
                # print "label:", label
                # print "channels:", channels
                # print "height:", height
                # print "width:", width


                # name = key.split("/")[-1].split(".")[0]
                # if name != "test_012260":
                #     continue

                image_x = np.fromstring(datum.data, dtype=np.uint8)      # 字符串转换为矩阵

                image = cv2.imdecode(image_x, -1)                       # decode

                #image = cv2.imread("/home/aiserver/result/word_dect/res_img/iou0_s23_195.jpg")
                img_size = image.shape

                crop_x = int(img_size[1] * bei_x)
                crop_y = int(img_size[0] * bei_y)


                #print(image.shape[0]//bei_x//2)
                if crop_y != 0:
                    image = image[int(crop_y/off_set):img_size[0]-int(crop_y*(off_set-1)/off_set) -1 ,]

                if crop_x != 0:
                    image = image[0:img_size[0]-1,
                            crop_x//2:img_size[1]-crop_x//2 -1 ]


                #img_size = image.shape
                #print(image)
                im = image.copy()
                for scale in scales:
                    # try:
                    image_resize_height = scale[0]
                    image_resize_width = scale[1]
                    transformer = caffe.io.Transformer({'data': (1, 3, image_resize_height, image_resize_width)})
                    transformer.set_transpose('data', (2, 0, 1))
                    transformer.set_mean('data', np.array([128, 128, 128]))  # mean pixel
                    #transformer.set_raw_scale('data',
                    #                          1)  # the reference model operates on images in [0,255] range instead of [0,1]
                    #transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

                    net.blobs['data'].reshape(1, 3, image_resize_height, image_resize_width)
                    transformed_image = transformer.preprocess('data', image)
                    #print transformed_image
                    net.blobs['data'].data[...] = transformed_image
                    #print(transformed_image)
                    # print(transformed_image)
                    # Forward pass.
                    detections = net.forward()['detection_out']
                    #print(detections)
                    # Parse the outputs.
                    det_label = detections[0, 0, :, 1]
                    det_conf = detections[0, 0, :, 2]
                    det_xmin = detections[0, 0, :, 3]
                    det_ymin = detections[0, 0, :, 4]
                    det_xmax = detections[0, 0, :, 5]
                    det_ymax = detections[0, 0, :, 6]

                    # Get detections with confidence higher than 0.1.
                    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0]

                    top_conf = det_conf[top_indices]
                    top_label_indices = det_label[top_indices].tolist()
                    top_xmin = det_xmin[top_indices]
                    top_ymin = det_ymin[top_indices]
                    top_xmax = det_xmax[top_indices]
                    top_ymax = det_ymax[top_indices]
                    idx_max = 0
                    xmin = ((top_xmin[idx_max] * image.shape[1]))
                    ymin = ((top_ymin[idx_max] * image.shape[0]))
                    xmax = ((top_xmax[idx_max] * image.shape[1]))
                    ymax = ((top_ymax[idx_max] * image.shape[0]))
                    xmin = max(1, xmin)
                    ymin = max(1, ymin)
                    xmax = min(image.shape[1] - 1, xmax)
                    ymax = min(image.shape[0] - 1, ymax)

                    Reframe = [xmin, ymin, xmax, ymax]

                    if crop_x != 0:
                        gt_xmin_f = gt_xmin - crop_x // 2
                        gt_xmax_f = gt_xmax - crop_x // 2
                    else:
                        gt_xmin_f = gt_xmin
                        gt_xmax_f = gt_xmax

                    if crop_y !=0:
                        gt_ymin_f = gt_ymin - int(crop_y / off_set)
                        gt_ymax_f = gt_ymax - int(crop_y / off_set)

                    else:
                        gt_ymin_f = gt_ymin
                        gt_ymax_f = gt_ymax

                    #print(Reframe, image.shape)
                    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 255, 0))
                    cv2.rectangle(image, (int(gt_xmin_f), int(gt_ymin_f)),
                                  (int(gt_xmax_f), int(gt_ymax_f)), color=(0, 0, 255))
                    score = top_conf[idx_max]

                    iouFa = IOU(Reframe, (gt_xmin_f, gt_ymin_f, gt_xmax_f, gt_ymax_f), False)

                   # print "iou:", iouFa, "..", iouFa
                    analyze_scores.append(score)
                    analyze_iou.append(iouFa)
                    #
                    #cv2.imwrite("/home/aiserver/result/word_dect/res_new/{}_{}_{}".format(iouFa*100, score, key.split("/")[-1]), image)
                    # cv2.imshow("image", image)  # 显示图片
                    # cv2.waitKey(0)

            print(bei_x, bei_y)
            res.writelines("{}_{}_{}\n".format(bei_x, bei_y,off_set))
            analyze_scores = np.array(analyze_scores)
            analyze_iou = np.array(analyze_iou)
            standards = np.arange(0, 1.1, 0.1)
            Scores = []
            for standard in standards:
                analyze_scores01 = np.where(analyze_scores > standard , 1, 0)
                analyze_iou01 = np.where(analyze_iou > 0.7, 1, 0)
                score = np.sum(analyze_iou01 * analyze_scores01) * 1.0 / idx
                Scores.append(np.sum(score) * 100)
            print(Scores)
            res.writelines("{}\n".format(Scores))
            Scores = []
            for standard in standards:
                analyze_scores01 = np.where(analyze_scores > standard, 1, 0)
                analyze_iou01 = np.where(analyze_iou > 0.6, 1, 0)
                score = np.sum(analyze_iou01 * analyze_scores01) * 1.0 / idx
                Scores.append(np.sum(score) * 100)
            print(Scores)
            res.writelines("{}\n".format(Scores))
            Scores = []
            for standard in standards:
                analyze_scores01 = np.where(analyze_scores > standard, 1, 0)
                analyze_iou01 = np.where(analyze_iou > 0.5, 1, 0)
                score = np.sum(analyze_iou01 * analyze_scores01) * 1.0 / idx
                Scores.append(np.sum(score) * 100)
            print(Scores)
            res.writelines("{}\n".format(Scores))

            res.close()
