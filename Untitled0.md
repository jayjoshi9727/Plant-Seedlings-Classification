```python
# This program is an implementation of a Machine learning CNN model to correctly classify plant species
# from their corresponding images

# Name: Jay Joshi, Dhruval Patel
# Project Name: Plant Seedlings Classification
# Course: Knowledge Discovery from Data Bases
# University of Regina ,Fall 2020
```


```python
# Requirements to run our code

# Python 3.6 or higher 
# TensorFlow 2.X
# Keras
# keras-Tuner
# Sklearn
# Matplotlib
# Pillow
# Numpy
# Open-cv
# Pandas
# Glob
# Math
```


```python
# Import all necessary Libraries
from glob import glob
import cv2
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator,img_to_array
from keras.preprocessing.image import load_img
import PIL.Image
import pickle
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import itertools

```


```python
# Mount your google colab notebook with google drive
from google.colab import drive
drive.mount("/content/gdrive")
```

    Mounted at /content/gdrive
    


```python
Scale_To = 70  # px to scale
seed = 7  # fixing random
```


```python
# Data Augmentaion for balancing Imbalanced data set

path = '/content/gdrive/My Drive/train/*/*.png' # selecting files from train data for augmentation
files = glob(path)

datagen = ImageDataGenerator(
        rotation_range=180,  # Rotating Images
        zoom_range = 0.1, # Random zoom  
        width_shift_range=0.1,  # Shift Images Horizontally in Random Manner
        height_shift_range=0.1, # Shift Images Vertically in Random Manner
        horizontal_flip=True,  # Flip images horizontally
        vertical_flip=True  # Flip images vertically
    ) 
for img in files:
  print(img)
  image=load_img(img) # Load Images from files
  x=img_to_array(image) # Convert Images to Array
  x=x.reshape((1,)+ x.shape) # Reshape images for data Augmentation
  count=0
  for batch in datagen.flow(x,batch_size=1,save_to_dir='/content/gdrive/My Drive/train/*/*.png',
                            save_prefix='images',
                             save_format='png'):
    count+=1
    if count>=8:
      break 



```


```python
path = '/content/gdrive/My Drive/train/*/*.png' 
files = glob(path)
print(len(files))

train_Img = []
train_Label = []
j = 1
num = len(files)
# print(num)

# Obtain images and resizing, obtain labels
for img in files:
    print(str(j) + "/" + str(num), end="\r")
    train_Img.append(cv2.resize(cv2.imread(img), (Scale_To, Scale_To)))  # Get image (with resizing)
    train_Label.append(img.split('/')[-2])  # Get image label (folder name)
    j += 1

train_Img = np.asarray(train_Img)  # Data set for training Images
train_Label = pd.DataFrame(train_Label)  # Data set for training Labels

```

    6611
    


```python
# Visualizing different plant species from training data set
for i in range(10):
    plt.subplot(2, 4, i + 1)
    plt.imshow(train_Img[i])
print(train_Label)
```

                          0
    0     Scentless Mayweed
    1     Scentless Mayweed
    2     Scentless Mayweed
    3     Scentless Mayweed
    4     Scentless Mayweed
    ...                 ...
    6606           Charlock
    6607           Charlock
    6608           Charlock
    6609           Charlock
    6610           Charlock
    
    [6611 rows x 1 columns]
    


![png](output_7_1.png)



```python
# Apply the following steps for images processing to perform masking and remove background noise
# Use gaussian blur for remove noise
# Convert color RGB space to HSV space
# Create mask
# Create boolean mask
# Apply boolean mask and getting image whithout background 

clear_Train_Img = []
examples = []; getEx = True
for img in train_Img:
    blur_Img = cv2.GaussianBlur(img, (5, 5), 0)   # Make gaussian blur
    hsv_Img = cv2.cvtColor(blur_Img, cv2.COLOR_BGR2HSV)  # Convert from RGB to HSV space
    lower_green = (25, 40, 50)
    upper_green = (75, 255, 255)
    mask = cv2.inRange(hsv_Img, lower_green, upper_green)  # Create mask for green plants within range
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    bMask = mask > 0  # Create boolean mask
    clear = np.zeros_like(img, np.uint8)  # Create empty image
    clear[bMask] = img[bMask] 
    
    clear_Train_Img.append(clear) # Remove Background
    
#     Visualizing step by step process of masking 
    if getEx:
        plt.subplot(2, 3, 1); plt.imshow(img)
        plt.subplot(2, 3, 2); plt.imshow(blurImg) 
        plt.subplot(2, 3, 3); plt.imshow(hsvImg)  
        plt.subplot(2, 3, 4); plt.imshow(mask)  
        plt.subplot(2, 3, 5); plt.imshow(bMask)  
        plt.subplot(2, 3, 6); plt.imshow(clear)  
        getEx = False

clear_Train_Img = np.asarray(clear_Train_Img)
```


![png](output_8_0.png)



```python
# Visualizing Images 

for i in range(10):
    plt.subplot(2, 4, i + 1)
    plt.imshow(clear_Train_Img[i])
```


![png](output_9_0.png)



```python
# Normalizing pixel values
clear_Train_Img = clear_Train_Img / 255
print(len(clearTrainImg))
```

    6611
    


```python
open_file = open('/content/gdrive/My Drive/lists', "wb")
pickle.dump(clear_Train_Img, open_file)
open_file.close()

```


```python
# One hot encoding and creating classes for each plant species and Data Visualization of Balanced
# Data Set
le = preprocessing.LabelEncoder()
le.fit(train_Label[0])
print("Classes: " + str(le.classes_))
encode_Train_Labels = le.transform(train_Label[0])

# Make Categorical Variables
clear_Train_Label = np_utils.to_categorical(encode_Train_Labels)
num_clases = clear_Train_Label.shape[1]
print("Number of classes: " + str(num_clases))

# Print 12 classification labels
trainLabel[0].value_counts().plot(kind='bar')
```

    Classes: ['Black-grass' 'Charlock' 'Cleavers' 'Common Chickweed' 'Common wheat'
     'Fat Hen' 'Loose Silky-bent' 'Maize' 'Scentless Mayweed'
     'Shepherds Purse' 'Small-flowered Cranesbill' 'Sugar beet']
    Number of classes: 12
    




    <matplotlib.axes._subplots.AxesSubplot at 0x7f92bf851048>




![png](output_12_2.png)



```python
open_file = open('/content/gdrive/My Drive/label_list', "wb")
pickle.dump(clear_Train_Label, open_file)
open_file.close()
```


```python
from sklearn.model_selection import train_test_split

train_X, test_X, train_Y, test_Y = train_test_split(clear_Train_Img, clear_Train_Label, 
                                                test_size=0.1, random_state=seed, 
                                                stratify = clearTrainLabel)
```


```python
# Writing nitiona model definition for Hyper parameter tuning so keras-tuner can randomly search 
# the best optimized parameter with higher validation accuracy

def build_model(hp): 
  model=keras.Sequential([
    keras.layers.Conv2D( # defining convolutional layer to find optimized filter size
     filters=hp.Int('conv_1_filter',min_value=32,max_value=256,step=16),
     kernel_size=hp.Choice('conv_1_kernel',values=[3,5]),
     activation='relu',
     input_shape=(70,70,3)  
    ),
    keras.layers.Conv2D(
     filters=hp.Int('conv_2_filter',min_value=32,max_value=256,step=16),
     kernel_size=hp.Choice('conv_2_kernel',values=[3,5]),
     activation='relu',
     input_shape=(70,70,3)
    ),
    keras.layers.Conv2D(
     filters=hp.Int('conv_3_filter',min_value=32,max_value=256,step=16),
     kernel_size=hp.Choice('conv_3_kernel',values=[3,5]),
     activation='relu',
     input_shape=(70,70,3) 
    ),
    keras.layers.Flatten(),
    keras.layers.Dense( # defining dense layer for number of neurons in hidden layer
        units=hp.Int('dense_1_units',min_value=32,max_value=256,step=16),
        activation='relu'
    ),
    keras.layers.Dense(
        units=hp.Int('dense_2_units',min_value=32,max_value=256,step=16),
        activation='relu'
    ),
    keras.layers.Dense(12,activation='softmax')
  ])

  model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning rate',values=[1e-2,1e-3])),
                loss='categorical_crossentropy',metrics=['accuracy'])
  
  return model
```


```python
# Tuning search for best CNN model
tuner_search=RandomSearch(build_model,objective='val_accuracy',max_trials=7,directory='/content/gdrive/My Drive',
                          project_name='Plant Seedlings New')

```


```python
# Running the tuner search process
tuner_search.search(clearTrainImg,clearTrainLabel,epochs=5,validation_split=0.1)
```

    Trial 7 Complete [00h 01m 20s]
    val_accuracy: 0.0
    
    Best val_accuracy So Far: 0.11329305171966553
    Total elapsed time: 00h 10m 03s
    INFO:tensorflow:Oracle triggered exit
    


```python
# Model summary with highest validation accuracy
model=tuner_search.get_best_models(num_models=1)[0]
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 66, 66, 176)       13376     
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 62, 62, 144)       633744    
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 60, 60, 176)       228272    
    _________________________________________________________________
    flatten (Flatten)            (None, 633600)            0         
    _________________________________________________________________
    dense (Dense)                (None, 160)               101376160 
    _________________________________________________________________
    dense_1 (Dense)              (None, 32)                5152      
    _________________________________________________________________
    dense_2 (Dense)              (None, 12)                396       
    =================================================================
    Total params: 102,257,100
    Trainable params: 102,257,100
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# Fit the model on training data and save model check points
filepath='My Drive/Model'
checkpoint_all = ModelCheckpoint(filepath,
                                 verbose=1, save_best_only=False, mode='max')
callbacks_list=[checkpoint_all]
model.fit(clear_Train_Img,clear_Train_Label,epochs=20,validation_split=0.1,initial_epoch=5,callbacks=callbacks_list)
```

    Epoch 6/20
      2/186 [..............................] - ETA: 18s - loss: 0.3384 - accuracy: 0.8750WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0383s vs `on_train_batch_end` time: 0.0702s). Check your callbacks.
    186/186 [==============================] - ETA: 0s - loss: 0.2871 - accuracy: 0.9027
    Epoch 00006: saving model to My Drive/Model
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/tracking/tracking.py:111: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/tracking/tracking.py:111: Layer.updates (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.
    INFO:tensorflow:Assets written to: My Drive/Model/assets
    186/186 [==============================] - 23s 126ms/step - loss: 0.2871 - accuracy: 0.9027 - val_loss: 25.6198 - val_accuracy: 0.0529
    Epoch 7/20
    186/186 [==============================] - ETA: 0s - loss: 0.1386 - accuracy: 0.9600
    Epoch 00007: saving model to My Drive/Model
    INFO:tensorflow:Assets written to: My Drive/Model/assets
    186/186 [==============================] - 26s 140ms/step - loss: 0.1386 - accuracy: 0.9600 - val_loss: 31.7010 - val_accuracy: 0.0408
    Epoch 8/20
    186/186 [==============================] - ETA: 0s - loss: 0.0616 - accuracy: 0.9818
    Epoch 00008: saving model to My Drive/Model
    INFO:tensorflow:Assets written to: My Drive/Model/assets
    186/186 [==============================] - 34s 182ms/step - loss: 0.0616 - accuracy: 0.9818 - val_loss: 40.1099 - val_accuracy: 0.0498
    Epoch 9/20
    186/186 [==============================] - ETA: 0s - loss: 0.0322 - accuracy: 0.9909
    Epoch 00009: saving model to My Drive/Model
    INFO:tensorflow:Assets written to: My Drive/Model/assets
    186/186 [==============================] - 27s 144ms/step - loss: 0.0322 - accuracy: 0.9909 - val_loss: 38.5713 - val_accuracy: 0.0423
    Epoch 10/20
    186/186 [==============================] - ETA: 0s - loss: 0.1025 - accuracy: 0.9702
    Epoch 00010: saving model to My Drive/Model
    INFO:tensorflow:Assets written to: My Drive/Model/assets
    186/186 [==============================] - 34s 182ms/step - loss: 0.1025 - accuracy: 0.9702 - val_loss: 34.2073 - val_accuracy: 0.0423
    Epoch 11/20
    186/186 [==============================] - ETA: 0s - loss: 0.0404 - accuracy: 0.9884
    Epoch 00011: saving model to My Drive/Model
    INFO:tensorflow:Assets written to: My Drive/Model/assets
    186/186 [==============================] - 31s 166ms/step - loss: 0.0404 - accuracy: 0.9884 - val_loss: 42.9938 - val_accuracy: 0.0438
    Epoch 12/20
    186/186 [==============================] - ETA: 0s - loss: 0.0532 - accuracy: 0.9857
    Epoch 00012: saving model to My Drive/Model
    INFO:tensorflow:Assets written to: My Drive/Model/assets
    186/186 [==============================] - 30s 163ms/step - loss: 0.0532 - accuracy: 0.9857 - val_loss: 40.4262 - val_accuracy: 0.0302
    Epoch 13/20
    186/186 [==============================] - ETA: 0s - loss: 0.0356 - accuracy: 0.9897
    Epoch 00013: saving model to My Drive/Model
    INFO:tensorflow:Assets written to: My Drive/Model/assets
    186/186 [==============================] - 28s 151ms/step - loss: 0.0356 - accuracy: 0.9897 - val_loss: 40.8317 - val_accuracy: 0.0438
    Epoch 14/20
    186/186 [==============================] - ETA: 0s - loss: 0.0193 - accuracy: 0.9953
    Epoch 00014: saving model to My Drive/Model
    INFO:tensorflow:Assets written to: My Drive/Model/assets
    186/186 [==============================] - 30s 162ms/step - loss: 0.0193 - accuracy: 0.9953 - val_loss: 41.4474 - val_accuracy: 0.0544
    Epoch 15/20
    186/186 [==============================] - ETA: 0s - loss: 0.0056 - accuracy: 0.9987
    Epoch 00015: saving model to My Drive/Model
    INFO:tensorflow:Assets written to: My Drive/Model/assets
    186/186 [==============================] - 33s 178ms/step - loss: 0.0056 - accuracy: 0.9987 - val_loss: 47.3673 - val_accuracy: 0.0544
    Epoch 16/20
    186/186 [==============================] - ETA: 0s - loss: 0.0013 - accuracy: 0.9998
    Epoch 00016: saving model to My Drive/Model
    INFO:tensorflow:Assets written to: My Drive/Model/assets
    186/186 [==============================] - 30s 159ms/step - loss: 0.0013 - accuracy: 0.9998 - val_loss: 47.7845 - val_accuracy: 0.0498
    Epoch 17/20
    186/186 [==============================] - ETA: 0s - loss: 0.0018 - accuracy: 0.9995
    Epoch 00017: saving model to My Drive/Model
    INFO:tensorflow:Assets written to: My Drive/Model/assets
    186/186 [==============================] - 31s 164ms/step - loss: 0.0018 - accuracy: 0.9995 - val_loss: 48.9505 - val_accuracy: 0.0468
    Epoch 18/20
    186/186 [==============================] - ETA: 0s - loss: 0.0012 - accuracy: 0.9997
    Epoch 00018: saving model to My Drive/Model
    INFO:tensorflow:Assets written to: My Drive/Model/assets
    186/186 [==============================] - 29s 155ms/step - loss: 0.0012 - accuracy: 0.9997 - val_loss: 49.6791 - val_accuracy: 0.0468
    Epoch 19/20
    186/186 [==============================] - ETA: 0s - loss: 7.9076e-04 - accuracy: 0.9998
    Epoch 00019: saving model to My Drive/Model
    INFO:tensorflow:Assets written to: My Drive/Model/assets
    186/186 [==============================] - 34s 180ms/step - loss: 7.9076e-04 - accuracy: 0.9998 - val_loss: 50.4343 - val_accuracy: 0.0468
    Epoch 20/20
    186/186 [==============================] - ETA: 0s - loss: 0.0020 - accuracy: 0.9997
    Epoch 00020: saving model to My Drive/Model
    INFO:tensorflow:Assets written to: My Drive/Model/assets
    186/186 [==============================] - 28s 151ms/step - loss: 0.0020 - accuracy: 0.9997 - val_loss: 51.0457 - val_accuracy: 0.0498
    




    <tensorflow.python.keras.callbacks.History at 0x7f92c911a748>




```python
# Evaluate model on train and validation data
print(model.evaluate(train_X, train_Y))  # Evaluate on train set
print(model.evaluate(test_X, test_Y))  # Evaluate on validation set
```

    186/186 [==============================] - 5s 25ms/step - loss: 5.0818 - accuracy: 0.9044
    [5.081840515136719, 0.904353678226471]
    21/21 [==============================] - 0s 23ms/step - loss: 5.3870 - accuracy: 0.9048
    [5.386959075927734, 0.9048338532447815]
    


```python
# Print confusion Matrix on basis of model evaluation
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    fig = plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict values from validation data
pred_Y = model.predict(test_X)
predY_Classes = np.argmax(pred_Y, axis = 1) 
true_Y = np.argmax(testY, axis = 1) 

confusionMTX = confusion_matrix(true_Y, predY_Classes) 

# Plot matrix for all classification Label
plot_confusion_matrix(confusionMTX, classes = le.classes_)
```


![png](output_21_0.png)



```python
# Classification report with precision,recall and f1-score of each classification label
plt.matshow(confusion_matrix(true_Y, predY_Classes))
print(classification_report(true_Y, predY_Classes.astype(int), 
                            target_names = le.classes_))
```

                               precision    recall  f1-score   support
    
                  Black-grass       1.00      0.92      0.96        65
                     Charlock       0.00      0.00      0.00        58
                     Cleavers       0.62      1.00      0.76        52
             Common Chickweed       1.00      1.00      1.00        61
                 Common wheat       0.95      1.00      0.98        41
                      Fat Hen       1.00      1.00      1.00        57
             Loose Silky-bent       0.96      1.00      0.98        66
                        Maize       0.98      1.00      0.99        41
            Scentless Mayweed       0.90      1.00      0.94        60
              Shepherds Purse       0.84      1.00      0.91        43
    Small-flowered Cranesbill       0.90      1.00      0.94        60
                   Sugar beet       0.95      1.00      0.97        58
    
                     accuracy                           0.90       662
                    macro avg       0.84      0.91      0.87       662
                 weighted avg       0.84      0.90      0.87       662
    
    

    /usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


![png](output_22_2.png)



```python
# Repeat the entire procedure for train data set
Scale_To = 70  # px to scale
seed = 7  # fixing random

path = '/content/gdrive/My Drive/test/*/*.png' 
files = glob(path)

test_Img = []
test_Label = []
j = 1
num = len(files)

# Obtain images and resizing, obtain labels
for img in files:
    print(str(j) + "/" + str(num), end="\r")
    testImg.append(cv2.resize(cv2.imread(img), (Scale_To, Scale_To)))  # Get image (with resizing)
    testLabel.append(img.split('/')[-2])  # Get image label (folder name)
    j += 1

test_Img = np.asarray(testImg)  # Train images set
test_Label = pd.DataFrame(testLabel)  # Train labels set

```

    


```python
for i in range(10):
    plt.subplot(2, 4, i + 1)
    plt.imshow(test_Img[i])
print(test_Label)
```

                                 0
    0                      Fat Hen
    1                      Fat Hen
    2                      Fat Hen
    3                      Fat Hen
    4                      Fat Hen
    ..                         ...
    378  Small-flowered Cranesbill
    379  Small-flowered Cranesbill
    380  Small-flowered Cranesbill
    381  Small-flowered Cranesbill
    382  Small-flowered Cranesbill
    
    [383 rows x 1 columns]
    


![png](output_24_1.png)



```python
clear_Test_Img = []
examples = []; getEx = True
for img in test_Img:
    blur_Img = cv2.GaussianBlur(img, (5, 5), 0)   
    hsv_Img = cv2.cvtColor(blur_Img, cv2.COLOR_BGR2HSV)
    lower_green = (25, 40, 50)
    upper_green = (75, 255, 255)
    mask = cv2.inRange(hsv_Img, lower_green, upper_green)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    bMask = mask > 0  
    clear = np.zeros_like(img, np.uint8)
    clear[bMask] = img[bMask]  
    
    clear_Test_Img.append(clear)  
    if getEx:
        plt.subplot(2, 3, 1); plt.imshow(img)
        plt.subplot(2, 3, 2); plt.imshow(blurImg)
        plt.subplot(2, 3, 3); plt.imshow(hsvImg)
        plt.subplot(2, 3, 4); plt.imshow(mask)  
        plt.subplot(2, 3, 5); plt.imshow(bMask)  
        plt.subplot(2, 3, 6); plt.imshow(clear)  
        getEx = False

clear_Test_Img = np.asarray(clear_Test_Img)
```


![png](output_25_0.png)



```python
for i in range(10):
    plt.subplot(2, 4, i + 1)
    plt.imshow(clear_Test_Img[i])
```


![png](output_26_0.png)



```python
# One hot encoding and creating classification labels
le = preprocessing.LabelEncoder()
le.fit(test_Label[0])
print("Classes: " + str(le.classes_))
encode_test_Labels = le.transform(testLabel[0])

clear_test_Label = np_utils.to_categorical(encode_test_Labels)
num_clases = clear_test_Label.shape[1]
print("Number of classes: " + str(num_clases))

# Plot of label types numbers
testLabel[0].value_counts().plot(kind='bar')
```

    Classes: ['Black-grass' 'Charlock' 'Cleavers' 'Common Chickweed' 'Common wheat'
     'Fat Hen' 'Loose Silky-bent' 'Maize' 'Scentless Mayweed'
     'Shepherds Purse' 'Small-flowered Cranesbill' 'Sugar beet']
    Number of classes: 12
    




    <matplotlib.axes._subplots.AxesSubplot at 0x7f925ae7bef0>




![png](output_27_2.png)



```python
open_file = open('/content/gdrive/My Drive/test_list', "wb")
pickle.dump(clearTestImg, open_file)
open_file.close()
```


```python
open_file = open('/content/gdrive/My Drive/test_label', "wb")
pickle.dump(cleartestLabel, open_file)
open_file.close()
```


```python
# Model evaluation on test data set
print(model.evaluate(clear_Test_Img, clear_test_Label))
```

    12/12 [==============================] - 1s 63ms/step - loss: 1167.6361 - accuracy: 0.8747
    [1167.6361083984375, 0.87467360496521]
    


```python
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    fig = plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
pred_Y = model.predict(clear_Test_Img)
predY_Classes = np.argmax(pred_Y, axis = 1) 
true_Y = np.argmax(clear_test_Label, axis = 1) 

# confusion matrix
confusionMTX = confusion_matrix(true_Y, predY_Classes) 

# plot the confusion matrix
plot_confusion_matrix(confusionMTX, classes = le.classes_)
```


![png](output_31_0.png)



```python

plt.matshow(confusion_matrix(true_Y, predY_Classes))
print(classification_report(true_Y, predY_Classes.astype(int), 
                            target_names = le.classes_))
```

                               precision    recall  f1-score   support
    
                  Black-grass       0.89      1.00      0.94        42
                     Charlock       0.00      0.00      0.00        32
                     Cleavers       0.68      1.00      0.81        40
             Common Chickweed       1.00      0.92      0.96        24
                 Common wheat       0.97      0.88      0.92        33
                      Fat Hen       1.00      0.97      0.98        33
             Loose Silky-bent       1.00      0.89      0.94        36
                        Maize       0.91      1.00      0.95        21
            Scentless Mayweed       0.75      1.00      0.86        39
              Shepherds Purse       0.89      1.00      0.94        25
    Small-flowered Cranesbill       0.94      1.00      0.97        33
                   Sugar beet       0.87      0.80      0.83        25
    
                     accuracy                           0.87       383
                    macro avg       0.83      0.87      0.84       383
                 weighted avg       0.82      0.87      0.84       383
    
    

    /usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


![png](output_32_2.png)



