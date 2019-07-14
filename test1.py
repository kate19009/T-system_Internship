from model.vggnet import VGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import load_model
from imutils import paths
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-m", "--model", required=True)
ap.add_argument("-l", "--label-bin", required=True)
args = vars(ap.parse_args())

print("[INFO] loading images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (64, 64))
	data.append(image)
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)


data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


lb = LabelBinarizer()
labels = lb.fit_transform(labels)

model = load_model("output/vggnet.model")
loss, acc = model.evaluate(data, labels)
print("loss: ", loss, "\nacc: ", acc)
