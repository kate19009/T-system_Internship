import pickle
import cv2

from keras.models import load_model


def predict_image(image_path):

	image = cv2.imread(image_path)
	output = image.copy()
	image = cv2.resize(image, (64, 64))

	image = image.astype("float") / 255.0

	image = image.reshape((1, image.shape[0], image.shape[1],
			image.shape[2]))


	model = load_model("output/vggnet.model")

	lb = pickle.loads(open("output/vggnet_lb.pickle", "rb").read())

	preds = model.predict(image)

	i = preds.argmax(axis=1)[0]
	label = lb.classes_[i]

	text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
	cv2.putText(output, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
		(0, 0, 255), 1)

	cv2.imshow("Image", output)
	cv2.waitKey(0)

