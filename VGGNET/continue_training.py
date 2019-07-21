# import the necessary packages

# this package is required to work with plot
import matplotlib
import matplotlib.pyplot as plt
# this package is required to create and execute arrays
import numpy as np
# this package is required to reorder images
import random
# this package is required to work with binary labels
import pickle
# this package is required to load and display images
import cv2
# this package is required to work with image paths
import os

# this package is required to create binary labels
from sklearn.preprocessing import LabelBinarizer
# this package is required to split data into training and test
from sklearn.model_selection import train_test_split

# this package is required to build a text report
# showing the main classification metrics
from sklearn.metrics import classification_report

# this function is required to create new images for better learning
from keras.preprocessing.image import ImageDataGenerator
# use SGD as optimizer
from keras.optimizers import SGD
# this function that loads the neural network model
from keras.models import load_model
# this package is required to work with image paths
from imutils import paths
matplotlib.use("Agg")

# initialize data for training
# (initial learning rate, number of epochs, batch size)

INIT_LR = 0.01
EPOCHS = 70


# initialize our Stochastic Gradient Descent optimizer
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)

def continue_training(dataset_path, epoch, bs):
	# this function loads data set from a specified path,
	# number of epochs, batch size
	# and continues to train the neural network

	print("[INFO] loading images...")

	# initialize the data and labels
	data = []
	labels = []

	# randomly shuffle images for better training
	imagePaths = sorted(list(paths.list_images(dataset_path)))
	random.seed(42)
	random.shuffle(imagePaths)

	# loop over the input images
	for imagePath in imagePaths:
		# load the image
		image = cv2.imread(imagePath)
		# resize it to 64x64
		image = cv2.resize(image, (64, 64))
		# put the image in the data list
		data.append(image)
		# take out label from image path
		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)

	# scale the pixel values from [0, 255] to [0, 1]
	# in array form
	data = np.array(data, dtype="float") / 255.0
	# convert labels list to NumPy array
	labels = np.array(labels)

	# split the data into training and testing set using 80% of
	# the data for training and 20% for testing
	(trainX, testX, trainY, testY) = train_test_split(data,
		labels, test_size=0.2, random_state=42)

	# convert the labels from integers to vectors
	# to one-hot encoding and creation a pickle file
	lb = LabelBinarizer()
	trainY = lb.fit_transform(trainY)
	testY = lb.transform(testY)

	# construct the image generator to increase the training sample
	aug = ImageDataGenerator(rotation_range=40, width_shift_range=0.2,
		height_shift_range=0.2, shear_range=0.3, zoom_range=0.4,
		horizontal_flip=True, fill_mode="nearest")

	# load model
	model = load_model("vggnet.model")


	print("[INFO] training network...")

	# train the network
	trained = model.fit_generator(aug.flow(trainX, trainY, batch_size=bs),
		validation_data=(testX, testY), steps_per_epoch=len(trainX) // bs,
		epochs=epoch)

	print("[INFO] evaluating network...")
	# evaluate the network
	predictions = model.predict(testX, batch_size=32)
	print(classification_report(testY.argmax(axis=1),
		predictions.argmax(axis=1), target_names=lb.classes_))

	print("[INFO] serializing network and label binarizer...")
	# save the model and labels to storage
	model.save("vggnet.model")
	f = open("vggnet_lb.pickle", "wb")
	f.write(pickle.dumps(lb))
	f.close()

	# plot the training loss and accuracy
	N = np.arange(0, EPOCHS)
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(N, trained.history["loss"], label="train_loss")
	plt.plot(N, trained.history["val_loss"], label="val_loss")
	plt.plot(N, trained.history["acc"], label="train_acc")
	plt.plot(N, trained.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()
	plt.savefig("vggnet_plot.png")
