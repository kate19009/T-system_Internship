import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os
import warnings
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import load_model
from imutils import paths
matplotlib.use("Agg")


def continue_training(dataset_path):
	warnings.filterwarnings("ignore", category=FutureWarning)
	print("[INFO] loading images...")
	data = []
	labels = []

	imagePaths = sorted(list(paths.list_images(dataset_path)))
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


	(trainX, testX, trainY, testY) = train_test_split(data,
		labels, test_size=0.2, random_state=42)


	lb = LabelBinarizer()
	trainY = lb.fit_transform(trainY)
	testY = lb.transform(testY)


	aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
		height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
		horizontal_flip=True, fill_mode="nearest")

	model = load_model("output/vggnet.model")
	INIT_LR = 0.1
	EPOCHS = 70
	BS = 32


	print("[INFO] training network...")
	opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)

	H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
		validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
		epochs=EPOCHS)


	print("[INFO] evaluating network...")
	predictions = model.predict(testX, batch_size=32)
	print(classification_report(testY.argmax(axis=1),
		predictions.argmax(axis=1), target_names=lb.classes_))



	print("[INFO] serializing network and label binarizer...")
	model.save("output/vggnet.model")
	f = open(" output/vggnet_lb.pickle", "wb")
	f.write(pickle.dumps(lb))
	f.close()


	N = np.arange(0, EPOCHS)
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(N, H.history["loss"], label="train_loss")
	plt.plot(N, H.history["val_loss"], label="val_loss")
	plt.plot(N, H.history["acc"], label="train_acc")
	plt.plot(N, H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()
	plt.savefig("vggnet_plot.png")
