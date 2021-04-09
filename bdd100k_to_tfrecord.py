import json
import time
import os

with open("bdd100k_labels_images_train.json") as read_file:
  bdd100k_data = json.load(read_file)

IMG_WIDTH = 1280
IMG_HEIGHT = 720

classes = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'traffic light', 'traffic sign', 'train', 'truck'] 
classes_id = {v:idx for idx, v in enumerate(classes)}

dataset = []

print("Processing...")

i = 0
for img in bdd100k_data:
  main_path = '/content/dataset/images/images/train/'
  filename = main_path + img['name']
  image_format = b'jpg'

  if not os.path.isfile('C:/Users/mkamalel/Documents/SFU/MSE 491/Project/cardata/bdd100k_images/bdd100k/images/100k/train/' + img['name']):
    continue

  objects = [obj for obj in img['labels'] if obj['category'] in classes]

  xmins = [obj['box2d']['x1'] / IMG_WIDTH for obj in objects]
  xmaxs = [obj['box2d']['x2'] / IMG_WIDTH for obj in objects]
  ymins = [obj['box2d']['y1'] / IMG_HEIGHT for obj in objects]
  ymaxs = [obj['box2d']['y2'] / IMG_HEIGHT for obj in objects]
  classes_text = [obj['category'] for obj in objects]
  classes_num = [classes_id[obj['category']] for obj in objects]

  image_data = {
    "filename" : filename,
    "id" : str(i),
    "height": IMG_HEIGHT,
    "width": IMG_WIDTH,
    "object" : {
      "count" : len(objects),
      "bbox" : {
        "xmin" : xmins,
        "xmax" : xmaxs,
        "ymin" : ymins,
        "ymax" : ymaxs,
        "label" : classes_num,
        "text" : classes_text,
      },
    }
  }
  objects = []
  dataset.append(image_data)

  i+=1

with open("bdd100k_tf_format.json", 'w') as f:
    json.dump(dataset, f)
