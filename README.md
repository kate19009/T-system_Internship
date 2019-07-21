# T-system_Internship

![](https://img.shields.io/github/license/pandao/editor.md.svg) 
# About
ControlsNN is a library containing a convolutional neural network that is capable of identifying various controls. It was based on a simplified VGGNet model with 3 convolutional layers. The library includes a trained model, as well as continue training and prediction functions. There are 6 controls that a neural network define: textfield, button, radiobutton, checkbox, slider and spinner.

The architecture of the neural network is shown in the picture:

![Architecture](https://github.com//AnneVR/T-systems_Internship/raw/master/Architecture.png)
# Installation
To use ControlsNN, you will need Python >= 3.3 and some additional libraries:
- Tensorflow 1.14.0
- Keras
- OpenCV
- Matplotlib
- Pickle
- Numpy 
 
Also need to `pip install` the wheel file:
```
$ pip install /path/to/ControlsNN-1.0-py3-none-any.whl
```
You can verify that ControlsNN has been successfully installed and start working with it.
 
# Basic usage
## Controls detection
To use this neural network, you need a png picture of any size with the image of one of the 6 controls: textfield, button, radiobutton, checkbox, slider and spinner.
 
For example, you can use this ‘slider.png’ picture, which depicts the slider:

![slider](https://github.com//AnneVR/T-systems_Internship/raw/master/slider.png)

Using the following commands in the code you can get the result of neural network prediction:
```python
  from ControlsNN import predict
  
  predict.predict_image('slider.png')
 ```
As a result of this function, you get the neural network prediction regarding this image. As you can see, the control that depicted was correctly identified:

![result](https://github.com//AnneVR/T-systems_Internship/raw/master/result.png)
 
## Continue training

This library provides continue training for the model using the continue_training function:
```python
   def continue_training(dataset_path, epoch, bs)
        # this function loads data set from a specified path,
        # number of epochs, batch size
        # and continues to train the neural network
 ``` 
Using the following commands you can continue to train the model:
```python
  from ControlsNN import continue_training
  
  continue_training.continue_training('test',1,32)
 ```  
# License
This project is licensed under the terms of the MIT license, see LICENSE.

