---
layout: post
title: Toy Robot: Toy Classification Using CNN
image: "/posts/ToyRobot_CoverImage.png"
tags: [Deep Learning, CNN, Data Science, Computer Vision, Transfer Learning, Python]
---

This project uses a Convolutional Neural Network to train a computer model to recognize distinct classes of toys on a bespoke, self-collected data set. After experimenting with various model architectures and training parameters, however, I used transfer learning - and the power of MobilenetV2 - to acheive a 100 percent test set accuracy. CNN is all about optimizing model architecture and training parameters, but building a successful model has much to do with the nature and quality of the data too. 

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Data Pipeline](#data-pipeline)
- [03. CNN Overview](#cnn-overview)
- [04. Baseline Network](#cnn-baseline)
- [05. Tackling Overfitting With Dropout](#cnn-dropout)
- [06. Image Augmentation](#cnn-augmentation)
- [07. Hyper-Parameter Tuning](#cnn-tuning)
- [08. Transfer Learning](#cnn-transfer-learning)
- [09. Overall Results Discussion](#cnn-results)
- [10. Next Steps & Growth](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

I want to build a robot that will pick up my kids' toys, AND put them in the correct bins! Because my four-year old is better at playing with toys than picking them up, and because I'm better at throwing them all in one basket than sorting them, I'll use Deep Learning techniques to train a computer to recognize a Brio from a Bananagram (among other toys).

I'll use my own images of my kids' toys as a unique, albiet limited, custom dataset, because I'll want to simulate the real-world scenarios of his toys scattered throughout the house. 

If this is successful and put into place on a larger scale, no parent will ever step on a Lego again! 

<br>
<br>

### Actions <a name="overview-actions"></a>

Thanks to the *Keras* Deep Learning library, most of the tasks and procedures for building a computer vision model are made easy in Python. On the other hand, while Keras provides the tools, understanding and optimizing the model's architecture and training parameters is not so easy. In my case, since I will be taking my own pictures, I also have to think hard about what constitutes good, useful data. The process of building this model, broken into subsections below, is as follows:

Generate Data: 
1. Take images - This became a trial/error/experiment process throughout this project. 
2. Structure the data into training, validation, and Test set folders.
3. Define the data flow parameters and set up data generator objects.

Iterative Model Building Method: 

4. I Start with a simple baseline model. The first network consists of:
    **Two convolutional layers**, each with 
    **32 filters** and subsequent 
    **Max Pooling** Layers, 
    A single **Dense (fully connected) layer** following flattening with 32 neurons 
    followed by an output layer for five toy classes. 
    I use the **RELU** activation function on all layers, 
    and I use the **'ADAM'** learning optimizer. 

Then I will add or refine aspects to try to improve its predictability:

5. Add a **Dropout** layer to reduce overfitting (which will be tweaked throughout). 
6. Add **Image Augmentation** to the data pipeline to increase variation in the training data, as well as address overfitting.
7. Add a Learning Rate Reducer to smooth convergence. 
8. Experiment with Layers and Filters: 
    a. First adding more convolutional layer
    b. Increasing the filters in convolutional layers
    c. Deceasing the filters in the convolutional layers
    d. Increasing the kernel size

9. Finally, I compare my network's results against a **Transfer Learning** model based on MobilenetV2, a powerful CNN model that uses some advanced layering techniques. 

<br>
<br>

### Results <a name="overview-results"></a>

The baseline network suffered badly from overfitting, but the addition of Dropout & Image Augmentation elimited this entirely (and also let to underfitting).

In terms of Classification Accuracy on the Test Set, we saw:

* Baseline Network: **74.7%**
* Baseline + Dropout: **81%**
* Baseline + Dropout + Image Augmentation: **64%**
* Baseline + Dropout + Image Augmentation + Learning Rate Reducer: **74%**
* Experiment 2 (32, 32, 64, 32): **78.7%**
* Experiment 4 (32, 64, 64 (kernel = 5x5), 32): **80%**
* MobilenetV2 base model: **100%**

The use of Transfer Learning with the MobilenetV2 base architecture was a bittersweet success. I wanted a more accurate model of my own, but its hard to argue with the efficiency and predictive power of a network that will predict my kids toys 100% of the time. 

<br>
<br>

### Growth/Next Steps <a name="overview-growth"></a>

The concept here is demonstrated, if not proven. We have shown that we can get very accurate predictions - that should this robot come to market, it will at least be able to accurately predict in what bins your childs' toys belong. 

I hold out that there is considerable room for improvement in my own self-built model. The experimental architectures that I tested here do not exhaust the possibilities. I can use the Keras Tuner to get an optimal network architecture. And/or, I can revisit my dataset, which is currently 'small' and is likely riddled with bias. 

My current working hypothesis is that: with such limited data (100 images in each of my 5 training sets), the model is extremely sensitive to bias. I'll explore this bias in the write up below. For now, suffice it to say that the other image datasets that I've seen appear to be produced in laboratory like conditions, with carefully controlled lighting and background. Although I was systematic in collecting the data for this project, my house (and my iphone camera) are far from laboratory conditions. 

<br>
<br>
___

# Data Overview  <a name="data-overview"></a>

Although my kid has dozens of types of toys, I began with a modest set of five classes of toys: 
* Bananagrams (a game for adults, which has become material for Teddy's garbage truck)
* Brios 
* Cars 
* Duplos (big legos for younger builders)
* Magnatiles

Problems: 

At first glance, these toys appear distinct enough, but when considering how an algorithm might think about it, there are plenty of challenges. Duplos are mostly made of building blocks, but there are plenty of figurines, animals, and other structures that belong in the same toy bin. Duplos have distinct circular connectors, but they are also square-shaped, like bananagrams and magnatiles, and they are made of solid colors, like magnatiles and Brio cars. Brios, likewise, come with both natural-colored wooden train tracks and multi-colored train cars, which have wheels like the car toy set. Cars and Bananagrams are relatively small, which makes capturing images of the same proportions as the other toys quite difficult. While Teddy does have hundreds of Duplos and Brios to photograph, there are limited numbers of cars and magnatiles, which means my training, validation, and test sets will have multiple (however different) images of the same objects. 

[IMAGES PLACEHOLDER] 

Solutions: 

To simplify, I removed the Duplo figurines and the Brio train cars from the sample population. After some trial and error, I also diversified and stratified the backgrounds for images in each toy class. Finally, I cropped most images so that the toy occupies the majority of the frame. Because of the limited number of toys in some classes, I separated the actual toys for the training, validation, and test set images.

I ended up with 145 images of each toy, separated as follows: 
* 100 training set images (500 total)
* 30 validation set images (150 total)
* 15 test set images (75 total)

<br>
![alt text](/img/posts/toy_collage.png "Toy Robot Image Samples")

<br>
For ease of use in Keras, my data folder structure first splits into training, validation, and test directories, and within each of those is split again into directories based upon the five toy classes.

Images in the folders are varying sizes, but will be fed into the data pipeline as 128 x 128 pixel images. 

___
<br>

# Data Pipeline  <a name="data-pipeline"></a>

Before building the network architecture and then training and testing it - I use Keras' Image Data Generator to set up a pipeline for our images to flow from my local hard-drive through the network.

In the code below, I will:

* Import the required packages for the baseline model
* Set up the parameters for the data pipeline
* Set up the image generators to process the images as they come in
* Set up the generator flow - specifying what we want to pass in for each iteration of training

<br>

```python

# import the required python libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# data flow parameters
training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 128
img_height = 128
num_channels = 3
num_classes = 5

# image generators
training_generator = ImageDataGenerator(rescale = 1./255)
validation_generator = ImageDataGenerator(rescale = 1./255)

# image flows
training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                                      target_size = (img_width, img_height),
                                                                      batch_size = batch_size,
                                                                      class_mode = 'categorical')

# because I moved the folders around once or twice during this project, I now keep this code in place to ensure that everything aligns. 
print(training_set.class_indices)
print(validation_set.class_indices)
{'bananagrams': 0, 'brios': 1, 'cars': 2, 'duplos': 3, 'magnatiles': 4}
{'bananagrams': 0, 'brios': 1, 'cars': 2, 'duplos': 3, 'magnatiles': 4}

```
<br>

Images are resized down to 128 x 128 pixels (of three RGB channels), and they will be input 32 at a time (batch size) for training. I have five toy classes (the labels will conveniently come from the folder names inside the training and validation sets). 

Raw pixel values (ranging between 0 and 255) are normalized to help gradient descent find an optimal solution more efficiently.
___
<br>

# Convolutional Neural Network Overview <a name="cnn-overview"></a>

Convolutional Neural Networks (CNN) are an adaptation of Artificial Neural Networks and are primarily used for image data tasks.

To a computer, an image is a three dimensional dataframe (or *tensor*), made of rows and columns of pixels (in our case 128x128), each with 3 'channels' for intensity values for colors (Red, Green, and Blue, hence RBG), that range from 0 to 255 (before normalizing). Thus, each pixel (all 1,024 of them in a 128x128 image) is a function of a 3x255 color value. 

A Convolutional Neural Network then tries to make sense of these values to make predictions about the image, or to predict what the image is of - here one of the five possible toy classes. Of course, the pixel values themselves are meaningless, they only make sense in relation to each other in spatial dimensions. The network tries to learn these relatinships - turning the patterns that it finds into *features* much like we do as humans. By learning to associate those feature-patterns with the class labels that it is provided, the network learns which features are meaningful for each class of object. 

**Convolution** is the process in which images are scanned over with filters that detect patterns. **Pooling** compresses these into more generalizable representations. This process helps reduce the problem space (turning the image into a smaller and smaller generalizable representation of features), it also helps reduce the network's sensitivy to minor changes, in other words to know that two images are of the same object, even though the images are not *exactly* the same.

CNN's consist of multiple convolutional layers (each made of any number of filters), and pooling layers, and dense layers that compress and generalize the data in the image so that it can ultimately be turned into a probability of belonging to one class of object or another. 

As a Convolutional Neural Network trains, it iteratively calculates how well it is predicting on class labels as **loss**. It then heads backward through the network in a process known as **Back Propagation** to update the paramaters within the network, trying to minimize error and improve predictive accuracy. Image can be sent through the network any number of times (or, *epochs*) during training. Over time, it learns to find a good mapping between the input data and the output classes.

There are many aspects of a CNN's architecture (combination of layers and filters) and learning parameters (Activation function, learning rate, image augmentation, etc.) that can be changed to affect a model's performance. Many of these will be discussed below. I liken it to machine with a control panel that contains a series of buttons and dials; all of which can be adjusted to optimize the big red dial at the end: predictive accuracy. 

___
<br>

# Baseline Network <a name="cnn-baseline"></a>

<br>

#### Network Architecture

The baseline network architecture is simple, and gives us a starting point to refine from. This network contains:
* **2 Convolutional Layers**, each with **32 filters** 
* each with subsequent **Max Pooling** Layers
* Flatten layer
* One **single Dense (Fully Connected) layer** with **32 neurons**
* Output layer for five class predictions.
* I use the **relu** activation function on all layers, and use the **adam** optimizer.

```python

# network architecture
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', input_shape = (img_width, img_height, num_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

# compile network
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# view network architecture
model.summary()

```
<br>
The output printed below shows us more clearly our baseline architecture:

```

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 128, 128, 32)      896       
_________________________________________________________________
activation (Activation)      (None, 128, 128, 32)      0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 64, 64, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 64, 32)        9248      
_________________________________________________________________
activation_1 (Activation)    (None, 64, 64, 32)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 32, 32)        0         
_________________________________________________________________
flatten (Flatten)            (None, 32768)             0         
_________________________________________________________________
dense (Dense)                (None, 32)                1048608   
_________________________________________________________________
activation_2 (Activation)    (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 198       
_________________________________________________________________
activation_3 (Activation)    (None, 6)                 0         
=================================================================
Total params: 1,058,950
Trainable params: 1,058,950
Non-trainable params: 0
_________________________________________________________________

```

<br>

#### Training The Network

With the data pipeline and network architecture in place, we're ready to train the model. 

In the below code I:

* Specify the number of epochs for training
* Set a location for the trained network to be saved (architecture & parameters)
* Set a *ModelCheckPoint* callback to save the best network at any point during training (based upon validation accuracy)
* Train the network and save the results to an object called *history*

```python

# training parameters
num_epochs = 50
model_filename = 'models/toy_robot_basic_v01.h5'

# callbacks
save_best_model = ModelCheckpoint(filepath = model_filename,
                                  monitor = 'val_accuracy',
                                  mode = 'max',
                                  verbose = 1,
                                  save_best_only = True)

# train the network
history = model.fit(x = training_set,
                    validation_data = validation_set,
                    batch_size = batch_size,
                    epochs = num_epochs,
                    callbacks = [save_best_model])

```
<br>
The ModelCheckpoint callback means that the *best* model is saved, in terms of validation set performance - from *any point* during training. That is, although I'm telling the network to train for 50 epochs, or 50 rounds of data, there is no guarantee that it will continue to find better weights and biases throughout those 50 rounds. Usually, in fact, it will find the best fit before it reaches the 50th epoch, even though it will continue to adjust parameters until I tell it to stop. So the ModelCheckpoint function ensures that we don't lose progress. 

<br>

#### Analysis Of Training Results

In addition to saving the *best* model (to model_filename), we can use the *history* object that we created to analyze the performance of the network epoch by epoch. In the following code, I'll plot the training and validation loss, and its classification accuracy. 

```python

import matplotlib.pyplot as plt

# plot validation results
fig, ax = plt.subplots(2, 1, figsize=(15,15))
ax[0].set_title('Loss')
ax[0].plot(history.epoch, history.history["loss"], label="Training Loss")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation Loss")
ax[1].set_title('Accuracy')
ax[1].plot(history.epoch, history.history["accuracy"], label="Training Accuracy")
ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation Accuracy")
ax[0].legend()
ax[1].legend()
plt.show()

# get best epoch performance for validation accuracy
max(history.history['val_accuracy'])

```

<br>

![alt text](/img/posts/Baseline_Train_Val_Metrics.png "Toy Robot Baseline Accuracy Plot")

<br>

These results are not great. In terms of validation accuracy (bottom orange line), the plot shows that the model learns quickly, but then plateaus by the 5th epoch. It also quickly learns to predict the training data. Reaching 100% accuracy by the 12th epoch. But 100% on the training data is not a good thing, if the validation accuracy does not keep pace. 

The more important pattern revealed by these graphs is the significant gap between performance on the training and validation sets. This gap means that the model is **over-fitting.**

That is, the network is learning the features of the training data *so well* that it cannot see very far beyond it. In other words, it is memorizing the training data, and failing to find the generalizable patterns that would allow it to recognize similar objects in the validation set. This is not good, because it means that in the real world, my Toy Robot will get confused if it sees a lego that doesn't perfectly match the images that it was trained on. I want the model to be able to *generalize* about what makes a lego a lego, so that it can recognize a previously unseen lego from a bananagram. 

In the following sections, I'll add features to the model that address the overfitting problem, attempting to close the gap between the training and validation accuracy scores. 

<br>

#### Performance On The Test Set

The model trains only on the training data, but the validation data does inform this training, because the model saves its progress (its weights and bias values) every time the validation set accuracy improves. To get a truly 'real world' taste of how the model peforms, we can use it to predict on images that it has not seen at all during training - the test set.

The model's predictive accuracy on the test set thus provides a good metric of how well the many iterations of our model performs relative to each other. 

In the code below, I will:

* Import the required packages for importing the test set images.
* Set up the parameters for the predictions.
* Load in the saved model file from training.
* Create a function for preprocessing the test set images in the same way that training and validation images were.
* Create a function for making predictions, returning both predicted class label, and predicted class probability.
* Iterate through our test set images, preprocessing each and passing to the network for prediction.
* Create a Pandas DataFrame to hold all prediction data.

```python

# import required packages
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
from os import listdir

# parameters for prediction
model_filename = 'models/toy_robot_basic_v01.h5'
img_width = 128
img_height = 128
labels_list = ['bananagrams', 'brios', 'cars', 'duplos', 'magnatiles']

# load model
model = load_model(model_filename)

# image pre-processing function
def preprocess_image(filepath):
    
    image = load_img(filepath, target_size = (img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = image * (1./255)
    
    return image

# image prediction function
def make_prediction(image):
    
    class_probs = model.predict(image)
    predicted_class = np.argmax(class_probs)
    predicted_label = labels_list[predicted_class]
    predicted_prob = class_probs[0][predicted_class]
    
    return predicted_label, predicted_prob

# loop through test data
source_dir = 'data/test/'
folder_names = ['bananagrams', 'brios', 'cars', 'duplos', 'magnatiles']
actual_labels = []
predicted_labels = []
predicted_probabilities = []
filenames = []

for folder in folder_names:
    
    images = listdir(source_dir + '/' + folder)
    
    for image in images:
        
        processed_image = preprocess_image(source_dir + '/' + folder + '/' + image)
        predicted_label, predicted_probability = make_prediction(processed_image)
        
        actual_labels.append(folder)
        predicted_labels.append(predicted_label)
        predicted_probabilities.append(predicted_probability)
        filenames.append(image)
        
# create dataframe to analyse
predictions_df = pd.DataFrame({"actual_label" : actual_labels,
                               "predicted_label" : predicted_labels,
                               "predicted_probability" : predicted_probabilities,
                               "filename" : filenames})

predictions_df['correct'] = np.where(predictions_df['actual_label'] == predictions_df['predicted_label'], 1, 0)

```
<br>

Thus we have a convenient, and very useful dataframe storing our prediction data (predictions_df). A small sample of those 75 rows looks like this: 

<br>

| **actual_label** | **predicted_label** | **predicted_probability** | **filename** | **correct** |
|---|---|---|---|---|
| bananagrams | bananagrams | 0.65868604 | IMG_7817.jpg | 1 |
| brios | brios | 0.99941015 | b1.jpg | 1 |
| cars | magnatiles | 0.99988043 | c119.jpg | 0 |
| duplos | bananagrams | 0.6900331 | IMG_8532.jpg | 0 |
| magnatiles | magnatiles | 0.994294 | IMG_8635.jpg | 1 |

<br>

This data can be used to calculate the test set classification accuracy (below). 
We can also use the data to figure out where and why the model struggled or failed, by: 
* creating a confusion matrix (below).
* using a Grad-CAM analysis (below). 

<br>

#### Test Set Classification Accuracy

To calculate test set classification accuracy:

```python

# overall test set accuracy
test_set_accuracy = predictions_df['correct'].sum() / len(predictions_df)
print(test_set_accuracy)

```
<br>

The baseline network gets **74.7% classification accuracy** on the test set.  This is the metric I'll be trying to improve in subsequent iterations. 

<br>

#### Test Set Confusion Matrix

Overall Classification Accuracy is useful, but it can obscure where and why the model struggled.

Maybe the network is predicting extremely well on Bananagrams, but it thinks that Magnatiles are Brios? 

A Confusion Matrix can show us these patterns, which I create using the predictions dataframe below.

```python

# confusion matrix 
confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'])
print(confusion_matrix)
# or, with percentages larger dataframes
# confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'], normalize = 'columns')
# print(confusion_matrix)

actual_label     bananagrams  brios  cars  duplos  magnatiles
predicted_label                                              
bananagrams               10      1     0       0           0
brios                      4     12     0       1           1
cars                       0      1    13       0           0
duplos                     0      0     0      13           6
magnatiles                 1      1     2       1           8

```

<br>

So, while overall our test set accuracy was ~ 75% - for each individual class we see:

* Bananagrams: 66.7%
* Brios: 80%
* Cars: 86.7%
* Duplos: 86.7%
* Magnatiles: 53.3%

Insightful! I honestly thought the Magnatiles would be the most recognizable, but here the model thinks a big portion of them are Duplos. Perhaps its not surprising since Magnatiles and Duplos share common features. They are both square/blocky and made of solid colors, which combine to form multi-color, multi-block shapes. 

But that is just me guessing! To really see what features the model is picking up on, use a grad-CAM analysis. 

___
<br>

#### Grad-CAM Analysis

Gradient-weighted Class Activation Mapping, or Grad-CAM, is a way to visualize what the model sees by overlaying the activated features from the last convolutional layer onto the actual image!

A heatmap is used to color-code the regions of the image that the model found most useful for classifying it as one thing or another. 

In the code below, I: 

* Find the name of the last convolutional layer (layers are named as saved in Keras as part of the model object) (Any convolutional layer can be used, but the last one should be the most meaningful).
* Set the image properties and directory paths (as we did when calling test images) (Any image can be analyzed).
* Define the Grad-CAM function to turn activated features into mappable objects (I use a script I found, not fully sure what "tape" and "GradientTape" refer to).
* Define a function to preprocess image(s) to analyze.
* Define a function to overlay the heat-map on the images
* Define a function to plot the image and the heatmap.

```python

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

os.chdir("C:/Dat_Sci/Data Projects/Toy Robot")

# Load the model
model_path = 'models/Toy_Robot_basic_v01.h5'
model = load_model(model_path)

# Print all layers in the model to inspect names and shapes
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.output.shape)

# Find the last Conv2D layer in the model
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

# Get the last conv layer name (this is what we will use for Grad-CAM)
last_conv_layer_name = find_last_conv_layer(model)
print("Last conv layer name:", last_conv_layer_name)

# Define which conv layers to use for Grad-CAM (here we just use the last conv layer)
conv_layers = [last_conv_layer_name]



# Define image properties and directories
img_size = (128, 128)  # Model input size
test_dir = "data/test"  # Directory with test images
output_dir = "grad_cam"  # Directory to save Grad-CAM images
os.makedirs(output_dir, exist_ok=True)

# --- Grad-CAM Function ---
def grad_cam(model, img_array, target_layer_name):
    """Compute Grad-CAM heatmap for a specified convolutional layer."""
    # Get the target conv layer from the model
    conv_layer = model.get_layer(target_layer_name)
    
    # Create a model that maps the input image to the conv layer output and predictions
    grad_model = tf.keras.models.Model(
        inputs=[model.input],
        outputs=[conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        # Get the predicted class index
        class_idx = tf.argmax(predictions[0])
        # Use the score of the predicted class as the loss
        loss = predictions[:, class_idx]
    
    # Compute gradients of the loss with respect to the conv layer output
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_output = conv_output[0]
    heatmap = np.mean(conv_output * pooled_grads, axis=-1)
    
    # Normalize the heatmap between 0 and 1
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    
    return heatmap

# --- Image Preprocessing ---
def preprocess_image(img_path):
    """Load and preprocess an image for the model."""
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img, img_array

def overlay_heatmap(heatmap, img_path, alpha=0.4):
    """Overlay the Grad-CAM heatmap on the original image."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend heatmap with the original image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img

# --- Apply Grad-CAM on Test Images ---
def run_grad_cam_on_test_set():
    """Run Grad-CAM on test images for specified convolutional layers."""
    for category in os.listdir(test_dir):
        category_path = os.path.join(test_dir, category)
        if not os.path.isdir(category_path):
            continue  # Skip non-directory files

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                # Preprocess the image
                original_img, img_array = preprocess_image(img_path)

                # Iterate over the chosen conv layers (here, just the last conv layer)
                for layer in conv_layers:
                    heatmap = grad_cam(model, img_array, target_layer_name=layer)
                    superimposed_img = overlay_heatmap(heatmap, img_path)

                    # Save the Grad-CAM output image
                    heatmap_path = os.path.join(output_dir, f"heatmap_{layer}_{category}_{img_name}")
                    cv2.imwrite(heatmap_path, superimposed_img)

                    # Display the original and Grad-CAM images side by side
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    ax[0].imshow(original_img)
                    ax[0].set_title(f"Original Image ({category})")
                    ax[0].axis("off")

                    ax[1].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
                    ax[1].set_title(f"Grad-CAM at {layer}")
                    ax[1].axis("off")

                    plt.show()

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# --- Run Grad-CAM on the Test Set ---
run_grad_cam_on_test_set()

```
These heat-maps offer two important insights, which I'll call *Bias insight* and *Depth insight*, 

Below I show a raw test set image alongside its corresponding grad-CAM image.

First, we can clearly see whether or not the model is picking up on the features that would distinguish one class of object from another. We see in the first one, an ideal scenario, in which the model seems to have honed in on the Brio track itself, distinguishing the shape and the parallel grooves as its distinguishing features. Below that, however, we see the opposite, where the model seems to have found the floor around the actual Bananagram as the important feature for classification. 

![alt text](/img/posts/heatmap_conv2d_36_brios_IMG_8744.png "Grad-CAM_Good Feature Detection")

![alt text](/img/posts/heatmap_conv2d_36_bananagrams_IMG_7823.png "Grad-CAM_Bad Feature Detection")

##### Bias Insight
Bias is a pervasive issue in CNN tasks. At a basic level, bias happens when a model learns to predict on something other than the features that distinguish separate classes. Whether or not the model predicted these first two images correctly is not exactly my concern here. We know that if a model focuses on train tracks it will have a better chance of predicting train tracks that it hasn't seen before, but if the model can't recognize a Bananagram from the floor, it probably does not know what class it belongs too, even if it does guess correctly. But if a model is biased, it is probably because it has learned to *correctly* predict on other features... features that are associated with the class in training, but not in real life. 

In fact, the model may well have predicted this particular Banagram correctly (I won't tell you). It may actually associate the floor (or perhaps the shape of its cut out), with the class Bananagram. If so, it is perhaps because the other backgrounds in the Bananagram training set include the same background, so when it sees wooden floor background it thinks that it is a Bananagram. If all my Bananagram images were taken against a wood floor background, and all the Duplos were taken against a white table backdrop, the model could correctly guess the class of each just by identifying the features of the background. But the issue is more complex than that. If four out of the five classes of images have a 50/50% split in wood floor vs. white table backgrounds, but one class, lets say Bananagrams has a 70/30% split, the model might still be biased towards that 70% background, because all other things being equal, it is at least 70% correct for one class of objects if it identifies the background alone as its defining feature. Further, if is has trouble identifying distiguishing features in the other classes, then it might still guess Bananagram every time it sees a woodfloor background. Finally, it may be that the proportion of backgrounds is the same in each class of images, but because Bananagrams are smaller, they take up less space in the frame, and therefore there is more (wood floor) background evident in the class Bananagrams than there is in other classes. 

I'll address bias as a cocern with my self-collected data set again in the conclusion.

<br>

##### Depth Insight
Secondly, we can use the grad-CAM images to compare what the model predicted correctly and what it missed! In the images below it looks like the network learned the features of the Magnatiles quite well. The heat-map looks like it is focused on the whole shape, as well as the screws, and the magnets inside the object. 

![alt text](/img/posts/magnatile_baseline_gradcam_correct.png "Grad-CAM Correct Classification")

![alt text](/img/posts/magnatiles_baseline_gradCAM1_.png "Grad-CAM Incorrect Classification")

However, while the top image was correctly identified as a Magnatile, the bottom image was Identified as a Duplo. This *may* be a bias issue, but it also seems likely, this time, that the model is simply failing to understand the difference between Magnatiles and Duplos and deep enough level. 

In subsequent iterations, I'll tackle the many issues identified above with various methods that should improve the model's overall performance. 

<br>

# Model Improvements

Rather than reproduce all of the text and discussion above for each of the subsequent iterations, I'll describe basic changes and performance metrics in the table below. Then, in the sections that follow, I'll discuss what additions are made, the rationalle behind them, and the result that matters: test set accuracy. 

| **Model** | **Changes Made** | **Validation Accuracy** | **Test Accuracy** |
|---|---|---|---|---|
| 1 | Baseline (see above) | 73.3% | 74.7% | 
| 2 | Add Dropout (0.5) | 76% | 81% | 
| 3 | Add Image Augmentation | 72% | 64% | 
| 4 | Adjust Learning Rate | 78.7% | 74% | 
| 5 | Add 3rd Convolutional Layer with 64 filters (CV1_32, CV2_32, CV3_64, Dense_32), Reduce Dropout (0.25) | 78.7% | 78.7% | 
| 6 | Reduce Filters in 1st Layer (CV1_16, CV2_32, CV3_64, Dense_32) | 73% | 72% |
| 7 | Increase Filters and kernel size in 3rd layer (CV1_32, CV2_64, CV3_64 (kernel size = 5x5), Dense_32) | 76.7%  | 80% |
| 8 | Add 4th Convolutional Layer | (CV1_32, CV2_32, CV3_32, CV4_64 (kernel size = 3x3), Dense_32) | 75.3% |  |
| 9 | Use MobilenetV2 base model | 98% | 100% | 


## Overcoming Overfitting With Dropout <a name="cnn-dropout"></a>

<br>

#### Dropout Overview

Dropout is a technique used in Deep Learning primarily to reduce the effects of over-fitting. 

As we have seen, *over-fitting* happens when the network learns the patterns of the training data so specifically that it essentially memorizes those images as the class of object itself. Then when it sees the same class of object in a different image (in the validation or test set), it cannot recognize it. 

*Dropout* is a technique in which, for each batch of observations that is sent forwards through the network, a pre-specified portion of the neurons in a hidden layer are randomly deactivated. This can be applied to any number of the hidden layers. 

When neurons are deactivated - they take no part in the passing of information through the network.

The math is the same, the network will process everything as it always would (taking the sum of the inputs multiplied by the weights, and adding a bias term, applying activation functions, and updating the network’s parameters using Back Propagation) - but now some of the neurons are simply turned off. If some neurons are turned off, then the other neurons have to jump in and pick up the slack (so to speak). If those other neurons were previously dedicated to certain very specific features of training images, they will now be forced to generalize a bit more. If over-trained neurons that were turned off in one epoch jump back in in the next, they now contend with a model that has found more generalizable patterns and will have to tune accordingly. 

Over time, with different combinations of neurons being ignored for each mini-batch of data - the network becomes more adept at generalising and thus is less likely to overfit to the training data. Since no particular neuron can rely on the presence of other neurons, and the features with which they represent - the network learns more robust features, and are less susceptible to noise.

<br>

#### Implementing Dropout

Adding dropout using Keras is as simple as installing the function and adding a line of code. I saved a new code sheet with the three following changes:

```python

# import
from tensorflow.keras.layers import Dropout

# add dropout to the output layer
# ... 
model.add(Dense(num_classes))
model.add(Dropout(0.5))
model.add(Activation('softmax')) 
# ...

# save model 
model_filename = 'models/toy_robot_dropout_v01.h5'

```
<br>

#### Results with Dropout

Other than the above, the following output results from the same exact code as our baseline model. Adding dropout was the only change. 

<br>

![alt text](/img/posts/DropoutModel_Train_Val_Metrics.png "Toy Robot Dropout Accuracy Plot")

<br>

The best classification accuracy on the *validation set* was **76%**, not significantly higher than the **75.3%** we saw for the baseline network. 
Validation set accuracy plateaus early again, at about the 10th epoch. 

Accuracy on the *test set* was **81%**, which is a nice bump from the **74.7%** test set accuracy from the baseline model. 

The model is no longer over-fitting. The gap between the classification accuracy on the training set and the validation set has been eliminated. In fact, the model is consistently predicting better on the validation set, which might indicate that the validation set data is more consistent within each class. 

On the other hand, we still see a divergence with respect to training vs. validation loss. This means that even though the network is consistently predicting the validation set at about 72-76%, it is become less confident in its predictions. That is the probabilities output associated with its predictions are likely going down. It is becoming less confident in the validation preditions, and more confident in the training predictions. 

Next, I turn to another method for reducing overfitting, Image Augmentation. 

___
<br>

# Image Augmentation <a name="cnn-augmentation"></a>

<br>

#### Image Augmentation Overview

Image Augmentation is a concept in Deep Learning that aims to not only increase predictive performance, but also to increase the robustness of the network through regularisation.

Instead of passing in each of the training set images as it stands, with Image Augmentation we pass in many transformed *versions* of each image.  This results in increased variation within our training data (without having to explicitly collect more images) meaning the network has a greater chance to understand and learn the objects we’re looking to classify, in a variety of scenarios.

Common transformation techniques are:

* Rotation
* Horizontal/Vertical Shift
* Shearing
* Zoom
* Horizontal/Vertical Flipping
* Brightness Alteration

When applying Image Augmentation using Keras' ImageDataGenerator class, we do this "on-the-fly" meaning the network does not actually train on the *original* training set image, but instead on the generated/transformed *versions* of the image - and this version changes each epoch.  In other words - for each epoch that the network is trained, each image will be called upon, and then randomly transformed based upon the specified parameters - and because of this variation, the network learns to generalise a lot better for many different scenarios.

<br>
#### Implementing Image Augmentation

We apply the Image Augmentation logic into the ImageDataGenerator class that exists within our Data Pipeline.

It is important to note is that we only ever do this for our training data, we don't apply any transformation on our validation or test sets.  The reason for this is that we want our validation & test data be static, and serve us better for measuring our performance over time.  If the images in these set kept changing because of transformations it would be really hard to understand if our network was actually improving, or if it was just a lucky set of validation set transformations that made it appear that is was performing better!

When setting up and training the baseline & Dropout networks - we used the ImageGenerator class for only one thing, to rescale the pixel values. Now we will add in the Image Augmentation parameters as well, meaning that as images flow into our network for training the transformations will be applied.

In the code below, we add these transformations in and specify the magnitudes that we want each applied:

```python

# image generators
training_generator = ImageDataGenerator(rescale = 1./255,
                                        rotation_range = 20,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        zoom_range = 0.1,
                                        horizontal_flip = True,
                                        brightness_range = (0.5,1.5),
                                        fill_mode = 'nearest')

validation_generator = ImageDataGenerator(rescale = 1./255)

```
<br>
We apply a **rotation_range** of 20.  This is the *degrees* of rotation, and it dictates the *maximum* amount of rotation that we want.  In other words, a rotation value will be randomly selected for each image, each epoch, between negative and positive 20 degrees, and whatever is selected, is what will be applied.

We apply a **width_shift_range** and a **height_shift_range** of 0.2.  These represent the fraction of the total width and height that we are happy to shift - in other words we're allowing Keras to shift our image *up to* 20% both vertically and horizonally.

We apply a **zoom_range** of 0.1, meaning a maximum of 10% inward or outward zoom.

We specify **horizontal_flip** to be True, meaning that each time an image flows in, there is a 50/50 chance of it being flipped.

We specify a **brightness_range** between 0.5 and 1.5 meaning our images can become brighter or darker.

Finally, we have **fill_mode** set to "nearest" which will mean that when images are shifted and/or rotated, we'll just use the *nearest pixel* to fill in any new pixels that are required - and it means our images still resemble the scene, generally speaking!

Again, it is important to note that these transformations are applied *only* to the training set, and not the validation set.

<br>
#### Updated Network Architecture

Our network will be the same as the baseline network.  We will not apply Dropout here to ensure we can understand the true impact of Image Augmentation for our task.

<br>
#### Training The Updated Network

We run the exact same code to train this updated network as we did for the baseline network (50 epochs) - the only change is that we modify the filename for the saved network to ensure we have all network files for comparison.

<br>
#### Analysis Of Training Results

As we again saved our training process to the *history* object, we can now analyse & plot the performance (Classification Accuracy, and Loss) of the updated network epoch by epoch.

With the baseline network we saw very strong overfitting in action - it will be interesting to see if the addition of Image Augmentation helps in the same way that Dropout did!

The below image shows the same two plots we analysed for the updated network, the first showing the epoch by epoch **Loss** for both the training set (blue) and the validation set (orange) & the second show the epoch by epoch **Classification Accuracy** again, for both the training set (blue) and the validation set (orange).

<br>
![alt text](/img/posts/cnn-augmentation-accuracy-plot.png "CNN Dropout Accuracy Plot")

<br>
Firstly, we can see a peak Classification Accuracy on the validation set of around **97%** which is higher than the **83%** we saw for the baseline network, and higher than the **89%** we saw for the network with Dropout added.

Secondly, and what we were again really looking to see, is that gap between the Classification Accuracy on the training set, and the validation set has been mostly eliminated. The two lines are trending up at more or less the same rate across all epochs of training - and the accuracy on the training set also never reach 100% as it did before meaning that Image Augmentation is also giving the network this *generalisation* that we want!

The reason for this is that the network is getting a slightly different version of each image each epoch during training, meaning that while it's learning features, it can't cling to a *single version* of those features!

<br>
#### Performance On The Test Set

During training, we assessed our updated networks performance on both the training set and the validation set.  Here, like we did for the baseline & Dropout networks, we will get a view of how well our network performs when predict on data that was *no part* of the training process whatsoever - our test set.

We run the exact same code as we did for the earlier networks, with the only change being to ensure we are loading in network file for the updated network

<br>
#### Test Set Classification Accuracy

Our baseline network achieved a **75% Classification Accuracy** on the test set, and our network with Dropout applied achieved **85%**.  With the addition of Image Augmentation we saw both a reduction in overfitting, and an increased *validation set* accuracy.  On the test set, we again see an increase vs. the baseline & Dropout, with a **93% Classification Accuracy**. 

<br>
#### Test Set Confusion Matrix

As mentioned above, while overall Classification Accuracy is very useful, but it can hide what is really going on with the network's predictions!

The standout insight for the baseline network was that Bananas has only a 20% Classification Accuracy, very frequently being confused with Lemons.  Dropout, through the additional *generalisation* forced upon the network, helped a lot - let's see how our network with Image Augmentation fares!

Running the same code from the baseline section on results for our updated network, we get the following output:

```

actual_label     apple  avocado  banana  kiwi  lemon  orange
predicted_label                                             
apple              0.9      0.0     0.0   0.0    0.0     0.0
avocado            0.0      1.0     0.0   0.0    0.0     0.0
banana             0.1      0.0     0.8   0.0    0.0     0.0
kiwi               0.0      0.0     0.0   0.9    0.0     0.0
lemon              0.0      0.0     0.2   0.0    1.0     0.0
orange             0.0      0.0     0.0   0.1    0.0     1.0

```
<br>
Along the top are our *actual* classes and down the side are our *predicted* classes - so counting *down* the columns we can get the Classification Accuracy (%) for each class, and we can see where it is getting confused.

So, while overall our test set accuracy was 93% - for each individual class we see:

* Apple: 90%
* Avocado: 100%
* Banana: 80%
* Kiwi: 90%
* Lemon: 100%
* Orange: 100%

All classes here are being predicted *more accurately* when compared to the baseline network, and *at least as accurate or better* when compared to the network with Dropout added.

Utilising Image Augmentation *and* applying Dropout will be a powerful combination!

___
<br>
# Hyper-Parameter Tuning <a name="cnn-tuning"></a>

<br>
#### Keras Tuner Overview

So far, with our Fruit Classification task, we have:

* Started with a baseline model
* Added Dropout to help with overfitting
* Utilised Image Augmentation

The addition of Dropout, and Image Augmentation boosted both performance and robustness - but there is one thing we've not tinkered with yet, and something that *could* have a big impact on how well the network learns to find and utilise important features for classifying our fruits - and that is the network *architecture*!

So far, we've just used 2 convolutional layers, each with 32 filters, and we've used a single Dense layer, also, just by coincidence, with 32 neurons - and we admitted that this was just a place to start, our baseline.

One way for us to figure out if there are *better* architectures, would be to just try different things. Maybe we just double our number of filters to 64, or maybe we keep the first convolutional layer at 32, but we increase the second to 64.Perhaps we put a whole lot of neurons in our hidden layer, and then, what about things like our use of Adam as an optimizer, is this the best one for our particular problem, or should we use something else?

As you can imagine, we could start testing all of these things, and noting down performances, but that would be quite messy.

Here we will instead utlise *Keras Tuner* which will make this a whole lot easier for us!

At a high level, with Keras Tuner, we will ask it to test, a whole host of different architecture and parameter options, based upon some specifications that we put in place.  It will go off and run some tests, and return us all sorts of interesting summary statistics, and of course information about what worked best.

Once we have this, we can then create that particular architecture, train the network just as we've always done - and analyse the performance against our original networks.

Our data pipeline will remain the same as it was when applying Image Augmentation.  The code below shows this, as well as the extra packages we need to load for Keras-Tuner.

```python

# import the required python libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
import os

# data flow parameters
training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 128
img_height = 128
num_channels = 3
num_classes = 6

# image generators
training_generator = ImageDataGenerator(rescale = 1./255)
validation_generator = ImageDataGenerator(rescale = 1./255)

# image flows
training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                                      target_size = (img_width, img_height),
                                                                      batch_size = batch_size,
                                                                      class_mode = 'categorical')

```

<br>
#### Application Of Keras Tuner

Here we specify what we want Keras Tuner to test, and how we want it to test it!

We put our network architecture into a *function* with a single parameter called *hp* (hyperparameter)

We then make use of several in-build bits of logic to specify what we want to test.  In the code below we test for:

* Convolutional Layer Count - Between 1 & 4
* Convolutional Layer Filter Count - Between 32 & 256 (Step Size 32)
* Dense Layer Count - Between 1 & 4
* Dense Layer Neuron Count - Between 32 & 256 (Step Size 32)
* Application Of Dropout - Yes or No
* Optimizer - Adam or RMSProp

```python

# network architecture
def build_model(hp):
    model = Sequential()
    
    model.add(Conv2D(filters = hp.Int("Input_Conv_Filters", min_value = 32, max_value = 256, step = 32), kernel_size = (3, 3), padding = 'same', input_shape = (img_width, img_height, num_channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    
    for i in range(hp.Int("n_Conv_Layers", min_value = 1, max_value = 3, step = 1)):
    
        model.add(Conv2D(filters = hp.Int(f"Conv_{i}_Filters", min_value = 32, max_value = 256, step = 32), kernel_size = (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D())
    
    model.add(Flatten())
    
    for j in range(hp.Int("n_Dense_Layers", min_value = 1, max_value = 4, step = 1)):
    
        model.add(Dense(hp.Int(f"Dense_{j}_Neurons", min_value = 32, max_value = 256, step = 32)))
        model.add(Activation('relu'))
        
        if hp.Boolean("Dropout"):
            model.add(Dropout(0.5))
    
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # compile network
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = hp.Choice("Optimizer", values = ['adam', 'RMSProp']),
                  metrics = ['accuracy'])
    
    return model

```
<br>
Once we have the testing logic in place - we use want to put in place the specifications for the search!

In the code below, we set parameters to:

* Point to the network *function* with the testing logic (hypermodel)
* Set the metric to optimise for (objective)
* Set the number of random network configurations to test (max_trials)
* Set the number of times to try each tested configuration (executions_per_trial)
* Set the details for the output of logging & results

```python

# search parameters
tuner = RandomSearch(hypermodel = build_model,
                     objective = 'val_accuracy',
                     max_trials = 30,
                     executions_per_trial = 2,
                     directory = os.path.normpath('C:/'),
                     project_name = 'fruit-cnn',
                     overwrite = True)

```
<br>
With the search parameters in place, we now want to put this into action.

In the below code, we:

* Specify the training & validation flows
* Specify the number of epochs for each tested configuration
* Specify the batch size for training

```python

# execute search
tuner.search(x = training_set,
             validation_data = validation_set,
             epochs = 40,
             batch_size = 32)

```
<br>
Depending on how many configurations are to be tested, how many epochs are required for each, and the speed of processing - this can take a long time, but the results will most definitely guide us towards a more optimal architecture!

<br>
#### Updated Network Architecture

Based upon the tested network architectures, the best in terms of validation accuracy was one that contains **3 Convolutional Layers**. The first has **96 filters** and the subsequent two each **64 filters**.  Each of these layers have an accompanying MaxPooling Layer (this wasn't tested). The network then has **1 Dense (Fully Connected) Layer** following flattening with **160 neurons** with **Dropout applied** - followed by our output layer. The chosen optimizer was **Adam**.

```python

# network architecture
model = Sequential()

model.add(Conv2D(filters = 96, kernel_size = (3, 3), padding = 'same', input_shape = (img_width, img_height, num_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(160))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

# compile network
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

```
<br>
The below shows us more clearly our optimised architecture:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_10 (Conv2D)           (None, 128, 128, 96)      2688      
_________________________________________________________________
activation_20 (Activation)   (None, 128, 128, 96)      0         
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 64, 64, 96)        0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 64, 64, 64)        55360     
_________________________________________________________________
activation_21 (Activation)   (None, 64, 64, 64)        0         
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 32, 32, 64)        0         
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 32, 32, 64)        36928     
_________________________________________________________________
activation_22 (Activation)   (None, 32, 32, 64)        0         
_________________________________________________________________
max_pooling2d_12 (MaxPooling (None, 16, 16, 64)        0         
_________________________________________________________________
flatten_5 (Flatten)          (None, 16384)             0         
_________________________________________________________________
dense_10 (Dense)             (None, 160)               2621600   
_________________________________________________________________
activation_23 (Activation)   (None, 160)               0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 160)               0         
_________________________________________________________________
dense_11 (Dense)             (None, 6)                 966       
_________________________________________________________________
activation_24 (Activation)   (None, 6)                 0         
=================================================================
Total params: 2,717,542
Trainable params: 2,717,542
Non-trainable params: 0
_________________________________________________________________

```

<br>
Our optimised architecture has a total of 2.7 million parameters, a step up from 1.1 million in the baseline architecture.

<br>
#### Training The Updated Network

We run the exact same code to train this updated network as we did for the baseline network (50 epochs) - the only change is that we modify the filename for the saved network to ensure we have all network files for comparison.

<br>
#### Analysis Of Training Results

As we again saved our training process to the *history* object, we can now analyse & plot the performance (Classification Accuracy, and Loss) of the updated network epoch by epoch.

The below image shows the same two plots we analysed for the tuned network, the first showing the epoch by epoch **Loss** for both the training set (blue) and the validation set (orange) & the second show the epoch by epoch **Classification Accuracy** again, for both the training set (blue) and the validation set (orange).

<br>
![alt text](/img/posts/cnn-tuned-accuracy-plot.png "CNN Tuned Accuracy Plot")

<br>
Firstly, we can see a peak Classification Accuracy on the validation set of around **98%** which is the highest we have seen from all networks so far, just higher than the 97% we saw for the addition of Image Augmentation to our baseline network.

As Dropout & Image Augmentation are in place here, we again see the elimination of overfitting.

<br>
#### Performance On The Test Set

During training, we assessed our updated networks performance on both the training set and the validation set.  Here, like we did for the baseline & Dropout networks, we will get a view of how well our network performs when predict on data that was *no part* of the training process whatsoever - our test set.

We run the exact same code as we did for the earlier networks, with the only change being to ensure we are loading in network file for the updated network

<br>
#### Test Set Classification Accuracy

Our optimised network, with both Dropout & Image Augmentation in place, scored **95%** on the Test Set, again marginally higher than what we had seen from the other networks so far.

<br>
#### Test Set Confusion Matrix

As mentioned each time, while overall Classification Accuracy is very useful, but it can hide what is really going on with the network's predictions!

Our 95% Test Set accuracy at an *overall* level tells us that we don't have too much to worry about here, but let's take a look anyway and see if anything interesting pops up.

Running the same code from the baseline section on results for our updated network, we get the following output:

```

actual_label     apple  avocado  banana  kiwi  lemon  orange
predicted_label                                             
apple              0.9      0.0     0.0   0.0    0.0     0.0
avocado            0.0      1.0     0.0   0.0    0.0     0.0
banana             0.0      0.0     0.9   0.0    0.0     0.0
kiwi               0.0      0.0     0.0   0.9    0.0     0.0
lemon              0.0      0.0     0.0   0.0    1.0     0.0
orange             0.0      0.0     0.0   0.1    0.0     1.0

```
<br>
Along the top are our *actual* classes and down the side are our *predicted* classes - so counting *down* the columns we can get the Classification Accuracy (%) for each class, and we can see where it is getting confused.

So, while overall our test set accuracy was 95% - for each individual class we see:

* Apple: 90%
* Avocado: 100%
* Banana: 90%
* Kiwi: 90%
* Lemon: 100%
* Orange: 100%

All classes here are being predicted *at least as accurate or better* when compared to the best network so far - so our optimised architecture does appear to have helped!

___
<br>
# Transfer Learning With VGG16 <a name="cnn-transfer-learning"></a>

<br>
#### Transfer Learning Overview

Transfer Learning is an extremely powerful way for us to utilise pre-built, and pre-trained networks, and apply these in a clever way to solve *our* specific Deep Learning based tasks.  It consists of taking features learned on one problem, and leveraging them on a new, similar problem!

For image based tasks this often means using all the the *pre-learned* features from a large network, so all of the convolutional filter values and feature maps, and instead of using it to predict what the network was originally designed for, piggybacking it, and training just the last part for some other task.

The hope is, that the features which have already been learned will be good enough to differentiate between our new classes, and we’ll save a whole lot of training time (and be able to utilise a network architecture that has potentially already been optimised).

For our Fruit Classification task we will be utilising a famous network known as **VGG16**.  This was designed back in 2014, but even by todays standards is a fairly heft network.  It was trained on the famous *ImageNet* dataset, with over a million images across one thousand different image classes. Everything from goldfish to cauliflowers to bottles of wine, to scuba divers!

<br>
![alt text](/img/posts/vgg16-architecture.png "VGG16 Architecture")

<br>
The VGG16 network won the 2014 ImageNet competition, meaning that it predicted more accurately than any other model on that set of images (although this has now been surpassed).

If we can get our hands on the fully trained VGG16 model object, built to differentiate between all of those one thousand different image classes, the features that are contained in the layer prior to flattening will be very rich, and could be very useful for predicting all sorts of other images too without having to (a) re-train this entire architecture, which would be computationally, very expensive or (b) having to come up with our very own complex architecture, which we know can take a lot of trial and error to get right!

All the hard work has been done, we just want to "transfer" those "learnings" to our own problem space.

<br>
#### Updated Data Pipeline

Our data pipeline will remain *mostly* the same as it was when applying our own custom built networks - but there are some subtle changes.  In the code below we need to import VGG16 and the custom preprocessing logic that it uses.  We also need to send our images in with the size 224 x 224 pixels as this is what VGG16 expects.  Otherwise, the logic stays as is.

```python

# import the required python libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# data flow parameters
training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 224
img_height = 224
num_channels = 3
num_classes = 6

# image generators
training_generator = ImageDataGenerator(preprocessing_function = preprocess_input,
                                        rotation_range = 20,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        zoom_range = 0.1,
                                        horizontal_flip = True,
                                        brightness_range = (0.5,1.5),
                                        fill_mode = 'nearest')
                                        
validation_generator = ImageDataGenerator(rescale = 1./255)

# image flows
training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                                      target_size = (img_width, img_height),
                                                                      batch_size = batch_size,
                                                                      class_mode = 'categorical')

```

<br>
#### Network Architecture

Keras makes the use of VGG16 very easy. We will download the *bottom* of the VGG16 network (everything up to the Dense Layers) and add in what we need to apply the *top* of the model to our fruit classes.

We then need to specify that we *do not* want the imported layers to be re-trained, we want their parameters values to be frozen.

The original VGG16 network architecture contains two massive Dense Layers near the end, each with 4096 neurons.  Since our task of classiying 6 types of fruit is more simplistic than the original 1000 ImageNet classes, we reduce this down and instead implement two Dense Layers with 128 neurons each, followed by our output layer.

```python

# network architecture
vgg = VGG16(input_shape = (img_width, img_height, num_channels), include_top = False)

# freeze all layers (they won't be updated during training)
for layer in vgg.layers:
    layer.trainable = False

flatten = Flatten()(vgg.output)

dense1 = Dense(128, activation = 'relu')(flatten)
dense2 = Dense(128, activation = 'relu')(dense1)

output = Dense(num_classes, activation = 'softmax')(dense2)

model = Model(inputs = vgg.inputs, outputs = output)

# compile network
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# view network architecture
model.summary()

```
<br>
The below shows us our final architecture:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten_7 (Flatten)          (None, 25088)             0         
_________________________________________________________________
dense_14 (Dense)             (None, 128)               3211392   
_________________________________________________________________
dense_15 (Dense)             (None, 128)               16512     
_________________________________________________________________
dense_16 (Dense)             (None, 6)                 774       
=================================================================
Total params: 17,943,366
Trainable params: 3,228,678
Non-trainable params: 14,714,688
_________________________________________________________________

```

<br>
Our VGG16 architecture has a total of 17.9 million parameters, much bigger than what we have built so far.  Of this, 14.7 million parameters are frozen, and 3.2 million parameters will be updated during each iteration of back-propagation, and these are going to be figuring out exactly how to use those frozen parameters that were learned from the ImageNet dataset, to predict our classes of fruit!

<br>
#### Training The Network

We run the exact same code to train this updated network as we did for the baseline network, although to start with for only 10 epochs as it is a much more computationally expensive training process.

<br>
#### Analysis Of Training Results

As we again saved our training process to the *history* object, we can now analyse & plot the performance (Classification Accuracy, and Loss) of the updated network epoch by epoch.

The below image shows the same two plots we analysed for the tuned network, the first showing the epoch by epoch **Loss** for both the training set (blue) and the validation set (orange) & the second show the epoch by epoch **Classification Accuracy** again, for both the training set (blue) and the validation set (orange).

<br>
![alt text](/img/posts/cnn-vgg16-accuracy-plot.png "VGG16 Accuracy Plot")

<br>
Firstly, we can see a peak Classification Accuracy on the validation set of around **98%** which is equal to the highest we have seen from all networks so far, but what is impressive is that it achieved this in only 10 epochs!

<br>
#### Performance On The Test Set

During training, we assessed our updated networks performance on both the training set and the validation set.  Here, like we did for all other networks, we will get a view of how well our network performs when predict on data that was *no part* of the training process whatsoever - our test set.

We run the exact same code as we did for the earlier networks, with the only change being to ensure we are loading in network file for the updated network

<br>
#### Test Set Classification Accuracy

Our VGG16 network scored **98%** on the Test Set, higher than that of our best custom network.

<br>
#### Test Set Confusion Matrix

As mentioned each time, while overall Classification Accuracy is very useful, but it can hide what is really going on with the network's predictions!

Our 98% Test Set accuracy at an *overall* level tells us that we don't have too much to worry about here, but for comparisons sake let's take a look!

Running the same code from the baseline section on results for our updated network, we get the following output:

```

actual_label     apple  avocado  banana  kiwi  lemon  orange
predicted_label                                             
apple              1.0      0.0     0.0   0.0    0.0     0.0
avocado            0.0      1.0     0.0   0.0    0.0     0.0
banana             0.0      0.0     1.0   0.0    0.0     0.0
kiwi               0.0      0.0     0.0   1.0    0.0     0.0
lemon              0.0      0.0     0.0   0.0    0.9     0.0
orange             0.0      0.0     0.0   0.0    0.1     1.0

```
<br>
Along the top are our *actual* classes and down the side are our *predicted* classes - so counting *down* the columns we can get the Classification Accuracy (%) for each class, and we can see where it is getting confused.

So, while overall our test set accuracy was 98% - for each individual class we see:

* Apple: 100%
* Avocado: 100%
* Banana: 100%
* Kiwi: 100%
* Lemon: 90%
* Orange: 100%

All classes here are being predicted *at least as accurate or better* when compared to the best custom network!

___
<br>
# Overall Results Discussion <a name="cnn-results"></a>

We have made some huge strides in terms of making our network's predictions more accurate, and more reliable on new data.

Our baseline network suffered badly from overfitting - the addition of both Dropout & Image Augmentation elimited this almost entirely.

In terms of Classification Accuracy on the Test Set, we saw:

* Baseline Network: **75%**
* Baseline + Dropout: **85%**
* Baseline + Image Augmentation: **93%**
* Optimised Architecture + Dropout + Image Augmentation: **95%**
* Transfer Learning Using VGG16: **98%**

Tuning the networks architecture with Keras-Tuner gave us a great boost, but was also very time intensive - however if this time investment results in improved accuracy then it is time well spent.

The use of Transfer Learning with the VGG16 architecture was also a great success, in only 10 epochs we were able to beat the performance of our smaller, custom networks which were training over 50 epochs.  From a business point of view we also need to consider the overheads of (a) storing the much larger VGG16 network file, and (b) any increased latency on inference.

___
<br>
# Growth & Next Steps <a name="growth-next-steps"></a>

The proof of concept was successful, we have shown that we can get very accurate predictions albeit on a small number of classes.  We need to showcase this to the client, discuss what it is that makes the network more robust, and then look to test our best networks on a larger array of classes.

Transfer Learning has been a big success, and was the best performing network in terms of classification accuracy on the Test Set - however we still only trained for a small number of epochs so we can push this even further.  It would be worthwhile testing other available pre-trained networks such as ResNet, Inception, and the DenseNet networks.
