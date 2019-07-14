
from keras.models import load_model
import argparse
import pickle
import cv2
import h5py


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
ap.add_argument("-m", "--model", required=True)
ap.add_argument("-l", "--label-bin", required=True)
ap.add_argument("-w", "--width", type=int, default=28)
ap.add_argument("-e", "--height", type=int, default=28)
ap.add_argument("-f", "--flatten", type=int, default=-1)
args = vars(ap.parse_args())


image = cv2.imread(args["image"])
output = image.copy()
image = cv2.resize(image, (args["width"], args["height"]))

image = image.astype("float") / 255.0


if args["flatten"] > 0:
	image = image.flatten()
	image = image.reshape((1, image.shape[0]))


else:
	image = image.reshape((1, image.shape[0], image.shape[1],
		image.shape[2]))


print("[INFO] loading network and label binarizer...")
model = load_model("output")

lb = pickle.loads(open(args["label_bin"], "rb").read())

preds = model.predict(image)

i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
	(0, 0, 255), 1)

cv2.imshow("Image", output)
cv2.waitKey(0)
