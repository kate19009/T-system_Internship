# import the necessary packages

# this package is required to work with binary labels
import pickle
# this package is required to load and display images
import cv2
# this function that loads the neural network model
from keras.models import load_model
# this package is required to load NN
import os


def predict_image(image_path):
    # this function loads an image from a specified path
    # and determines which control is shown

    # load the image and resize it
    image = cv2.imread(image_path)
    output = image.copy()
    image = cv2.resize(image, (64, 64))

    # scale the pixel values from [0, 255] to [0, 1]
    image = image.astype("float") / 255.0

    # add the batch dimension
    image = image.reshape((1, image.shape[0], image.shape[1],
            image.shape[2]))

    # load model and label binarizer
    model = load_model(os.path.abspath("venv/lib/python3.7/site-packages/ControlsNN/vggnet.model"))
    lb = pickle.loads(open(os.path.abspath("venv/lib/python3.7/site-packages/ControlsNN/vggnet_lb.pickle"), "rb").read())

    # make a prediction of what is depicted in the image
    preds = model.predict(image)

    # find the most exact match with the data
    # on which the neural network was trained
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]

    #show the image with the accuracy defined by the neural network
    text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
    cv2.putText(output, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (0, 0, 255), 1)
    cv2.imshow("Image", output)
    cv2.waitKey(0)

