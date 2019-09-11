import cv2
import numpy as np
import glob
import os
import xml.etree.ElementTree as ET

import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(get_session())

# Can change this to change the model used for inferencing
model_path = "semitrained_models/*.h5"
model = models.load_model(model_path, backbone_name="resnet50")
# Strings associated with each label index
labels_to_names = {0: "bike", 1: "non-bike"}
# Output path information
output_path = "output_imgs"

# Get paths of all input images
images_to_label = glob.glob("input_imgs/*.jpg")
images_to_label_png = glob.glob("input_imgs/*.png")
images_to_label.extend(images_to_label_png)

# Iterate through all input images and run inference on them
for image_path in images_to_label:
    image_name = os.path.basename(image)
    image = cv2.imread(image_path)

    # Padding image to make it square
    rows = image.shape[0]
    cols = image.shape[1]
    if rows < cols:
        padding = int((cols - rows) / 2)
        image = cv2.copyMakeBorder(image, padding, padding, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    elif rows > cols:
        padding = int((rows - cols) / 2)
        image = cv2.copyMakeBorder(image, 0, 0, padding, padding, cv2.BORDER_CONSTANT, (0, 0, 0))
    og_image = image.copy()

    # Preprocessing, before inputting into the model
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # Running the inferencing
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    # Correcting the box scale
    boxes /= scale

    xml_obj = ET.Element("annotation")
    folder = ET.Element("folder")
    folder.text = output_path
    xml_obj.append(folder)

    filename = ET.Element("filename")
    filename.text = image_name
    xml_obj.append(filename)

    path = ET.Element("path")
    path.text = os.path.abspath(output_path + "/" + image_name)
    xml_obj.append(path)

    source = ET.Element("source")
    database = ET.Element("database")
    database.text = "Unknown"
    source.append(database)
    xml_obj.append(source)

    size = ET.Element("size")
    width = ET.Element("width")
    width.text = str(cols)
    size.append(width)
    height = ET.Element("height")
    height.text = str(rows)
    size.append(height)
    depth = ET.Element("depth")
    depth.text = str(image.shape[2])
    size.append(depth)
    xml_obj.append(size)

    segmented = ET.Element("segmented")
    segmented.text = "0"
    xml_obj.append(segmented)

    # Visualizing detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < 0.5: # Change score threshold to display lower probability detections
            break # We can break because scores are sorted
        
        obj = ET.Element("object")
        name = ET.Element("name")
        name.text = labels_to_names[label]
        obj.append(name)

        pose = ET.Element("pose")
        pose.text = "Unspecified"
        obj.append(pose)

        truncated = ET.Element("truncated")
        truncated.text = "0"
        obj.append(truncated)

        difficult = ET.Element("difficult")
        difficult.text = "0"
        obj.append(difficult)

        bndbox = ET.Element("bndbox")
        xmin = ET.Element("xmin")
        xmin.text = str(box[0])
        bndbox.append(xmin)
        ymin = ET.Element("ymin")
        ymin.text = str(box[1])
        bndbox.append(ymin)
        xmax = ET.Element("xmax")
        xmax.text = str(box[2])
        bndbox.append(xmax)
        ymax = ET.Element("ymax")
        ymax.text = str(box[3])
        bndbox.append(ymax)
        obj.append(bndbox)

        xml_obj.append(obj)

    # Saving image with detections
    cur_image += 1
    cv2.imwrite(output_path + "/" + image_name, og_image)
    output_str = ET.tostring(xml_obj)

    xml_file = open(output_path + "/" + image_name[:-3] + ".xml", "w+")
    xml_file.write(output_str)    
    xml_file.close()

    print("Annotated " + str(cur_image) + " out of " + str(len(images_to_label)) + " images.")

print("Semiauto annotations completed.")