import numpy as np
import matplotlib.pyplot as plt
import glob
import cPickle
from PIL import Image, ImageFont, ImageDraw
import copy
import time
import json
import random

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe_root = '../..'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(2)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

voc_labelmap_file = '/home/siyu/dataset/coco/labelmap_coco-person.prototxt'
file = open(voc_labelmap_file, 'r')
voc_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), voc_labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

# model_def = 'D:\\v-sij\\COMPILE_SUCCESS_SSD\\caffe-windows\\models\\VGGNet\\VID\\SSD_500x500\\0804_lr_5e-4\\deploy.prototxt'
model_def = '/home/siyu/ssd-dev/part-ssd/jobs/VGGNet/ssd_coco_part_0.1/deploy.prototxt'
model_weights = '/home/siyu/ssd-dev/part-ssd/models/VGGNet/ssd_coco_part_0.1/VGG_ssd_coco_part_0.1_iter_150000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

image_resize = 512
net.blobs['data'].reshape(1,3,image_resize,image_resize)

data_root_path = '/home/siyu/dataset/coco/Val2014/JPEGImages'
result_root_path = '/home/siyu/detection_results/coco/ssd_coco_part_0.1_150000'
# subdirs = os.listdir(data_root_path)

if not os.path.isdir(result_root_path):
    os.makedirs(result_root_path)

# for s in range(len(subdirs)):
# mat_result_path = os.path.join(result_root_path, subdirs[s])
mat_result_path = result_root_path
mat_result_path = os.path.join(mat_result_path, 'video_result.pkl')

# if not os.path.isdir(os.path.join(result_root_path, subdirs[s])):
#     os.mkdir(os.path.join(result_root_path, subdirs[s]))


images = glob.glob(os.path.join(data_root_path, '*.jpg'))
#video_detections = [0]*len(images)
video_detections = []
cnt = 0 
total_time = 0

json_result = list()

for image_path in images:
    
    # draw = (cnt % 100 == 0)
    draw = True

    if cnt % 1000 == 0:
        print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ("finished %d images" % cnt)

    json_img_id = int(image_path.split("COCO_val2014_")[1].split(".jpg")[0])

    image = caffe.io.load_image(image_path)
    
    start = time.clock()
    
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    # Forward pass.
    detections = net.forward()['detection_out']

    finish = time.clock()

    # print image_path, (finish - start)
    total_time += (finish - start)
    # video_detections.append(copy.copy(detections))

    cnt = cnt + 1
    # Parse the outputs.
    det_label = detections[0,0,:,1]
    # print det_label
    det_conf = detections[0,0,:,2]
    # print det_conf
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]
    # det_clean = detections[0,0,:,7]
    # det_prob = detections[0,0,:,8:39]

    # print detections.shape

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.01]
    # print top_indices
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    # print top_label_indices
    top_labels = get_labelname(voc_labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    # top_clean = det_clean[top_indices]

    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if draw:
        plt.imshow(image)
        currentAxis = plt.gca()

    for i in xrange(top_conf.shape[0]):
        score = top_conf[i]
        # clean = top_clean[i]

        if draw and score >= 0.2:
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
                
            label = top_labels[i]
            name = 's%.2f'%(score)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            # color = colors[i % len(colors)]
            color = (random.random(), random.random(), random.random())
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=3))

            if i%4==0:
                currentAxis.text(xmin, ymin, name, bbox={'facecolor':'white', 'alpha':0.5})
            elif i%4==1:
                currentAxis.text(xmin, ymax, name, bbox={'facecolor':'white', 'alpha':0.5})
            elif i%4==2:
                currentAxis.text(xmax, ymax, name, bbox={'facecolor':'white', 'alpha':0.5})
            else:
                currentAxis.text(xmax, ymin, name, bbox={'facecolor':'white', 'alpha':0.5})

        x_res = round(top_xmin[i] * image.shape[1], 2)
        y_res = round(top_ymin[i] * image.shape[0], 2)
        w_res = round(top_xmax[i] * image.shape[1] - top_xmin[i] * image.shape[1], 2)
        h_res = round(top_ymax[i] * image.shape[0] - top_ymin[i] * image.shape[0], 2)

        json_cat_id = int(top_label_indices[i])
        json_bbox = [x_res, y_res, w_res, h_res]
        json_score = round(score, 3)
        # json_clean = round(clean, 3)

        json_per_result = [{"image_id": json_img_id, "category_id": json_cat_id, \
                            "bbox": json_bbox, "score": json_score}]
        json_result += json_per_result

    if draw:
        save_result_path = os.path.join(os.path.join(result_root_path, os.path.basename(image_path)))
        plt.savefig(save_result_path, bbox_inches='tight', pad_inches=0)
        plt.close()

# f = open(mat_result_path, 'w')
# cPickle.dump(video_detections, f)
# f.close()

# print json_result
json_final = json.dumps(json_result, sort_keys=False, separators=(',',':'))
f_json = open(os.path.join(result_root_path, 'result.json'), 'w')
f_json.write(json_final)
f_json.close()

print cnt
print "mean:", total_time/cnt

