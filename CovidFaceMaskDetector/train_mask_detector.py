import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tensorflow as tf
import keras
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.models import Sequential,load_model,save_model,Model
from keras.layers import Dense,Conv2D,Flatten,AveragePooling2D,Dropout,Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input
from keras.utils import to_categorical
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from glob import glob
from imutils import paths
# import warnings
import silence_tensorflow.auto
# warnings.filterwarnings("ignore", category=DeprecationWarning)

DATASET_DIR = "D:\Digant\Codes\Machine Learning\OnlineGitHub\observations-master\experiements\data"

INIT_LR = 1e-4
EPOCHS = 20
BS = 32


data_gen = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest",
    validation_split=0.2,
    preprocessing_function=preprocess_input)

train_generator = data_gen.flow_from_directory(
    DATASET_DIR,
    target_size=(224,224),
    batch_size=BS,
    class_mode='categorical',
    shuffle=True,
    subset='training') # set as training data

validation_generator = data_gen.flow_from_directory(
    DATASET_DIR, # same directory as training data
    target_size=(224,224),
    batch_size=BS,
    class_mode='categorical',
    shuffle=True,
    subset='validation') # set as validation data





base_model = MobileNetV2(weights='imagenet', 
                         include_top=False, 
                         input_tensor=Input(shape=(224,224,3))
                        )

head_model = base_model.output

head_model = AveragePooling2D(pool_size=(7,7))(head_model)
head_model = Flatten()(head_model)
head_model = Dense(128,activation='relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation='softmax')(head_model)

model = Model(inputs=base_model.inputs, outputs= head_model)


for layer in base_model.layers:
    layer.trainable = False
    
optimizer = Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)

model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

H = model.fit_generator(train_generator,
              steps_per_epoch=train_generator.samples // BS,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples// BS,
              epochs=EPOCHS
              )


model.save("mask_detector.h5")
# Evaluating the model
import pickle 

print("Now Saving")
pickle_out = open(f"fit_variable.pkl", "wb")
pickle.dump(H, pickle_out)
pickle_out.close()
filenames = test_generator.filenames
nb_samples = len(filenames)
y_pred = model.predict_generator(validation_generator,use_multiprocessing=True,steps=nb_samples)

# NEED to check if below line is needed.
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
y_pred = np.argmax(y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['with_mask', 'without_mask']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))





N = EPOCHS


plt.style.use("ggplot")

plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")


plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("LossAndAccuracy")
plt.show()

