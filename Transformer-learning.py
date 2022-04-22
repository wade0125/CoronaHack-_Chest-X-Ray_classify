import numpy as np 
import pandas as pd 
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input


summary = pd.read_csv('C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/Chest_xray_Corona_dataset_Summary.csv')
df = pd.read_csv('C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/Chest_xray_Corona_Metadata.csv')

replace_dict = {'Pnemonia':1,'Normal':0}
df['Label'] = df['Label'].replace(replace_dict)
train_df = df[df.Dataset_type=='TRAIN']
test_df = df[df.Dataset_type=='TEST']


training_data_path = 'C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'
testing_data_path = 'C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test'


# In[4]:


# Funtions for Making nd Removing subdirectories
def create_dir():
    try:
        os.makedirs('C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/train/Pneumonia')
        os.makedirs('C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/train/Normal')
        os.makedirs('C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/val/Pneumonia')
        os.makedirs('C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/val/Normal')
        os.makedirs('C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/test/Pneumonia')
        os.makedirs('C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/test/Normal')
    except:
        pass
def remove_dir():
    try:
        shutil.rmtree('C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/train')
        shutil.rmtree('C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/test')    
    except:
        pass


# In[5]:


# Seperate dataframes for different labels in test and train
train_pneumonia_df = train_df[train_df.Label==1]
train_normal_df = train_df[train_df.Label==0]
test_pneumonia_df = test_df[test_df.Label==1]
test_normal_df = test_df[test_df.Label==0]


# In[6]:


ntrain_p = len(train_pneumonia_df)
ntrain_n = len(train_normal_df)
tntrain = ntrain_p+ntrain_n

#Take 10% from train to be validation

nval_p = round(0.1*ntrain_p)
nval_n = round(0.1*ntrain_n)

print(nval_p)
print(nval_n)


# In[7]:


val_pneumonia_df = train_pneumonia_df[0:nval_p]
train_pneumonia_df = train_pneumonia_df[nval_p:]

val_normal_df = train_normal_df[0:nval_n]
train_normal_df = train_normal_df[nval_n:]


# In[8]:


# Copying the files to newly created locations. You may use Flow from dataframe attribute and skip all these steps. But I prefer to use flow from directory 
remove_dir()
create_dir()

training_images_pneumonia = train_pneumonia_df.X_ray_image_name.values.tolist()
training_images_normal = train_normal_df.X_ray_image_name.values.tolist()

val_images_pneumonia = val_pneumonia_df.X_ray_image_name.values.tolist()
val_images_normal = val_normal_df.X_ray_image_name.values.tolist()

testing_images_pneumonia = test_pneumonia_df.X_ray_image_name.values.tolist()
testing_images_normal = test_normal_df.X_ray_image_name.values.tolist()

for image in training_images_pneumonia:
    train_image_pneumonia = os.path.join(training_data_path, str(image))
    shutil.copy(train_image_pneumonia, 'C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/train/Pneumonia')
    
for image in training_images_normal:
    train_image_normal = os.path.join(training_data_path, str(image))
    shutil.copy(train_image_normal, 'C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/train/Normal')
    
for image in val_images_pneumonia:
    val_image_pneumonia = os.path.join(training_data_path, str(image))
    shutil.copy(val_image_pneumonia, 'C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/val/Pneumonia')
    
for image in val_images_normal:
    val_image_normal = os.path.join(training_data_path, str(image))
    shutil.copy(val_image_normal, 'C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/val/Normal')
    
for image in testing_images_pneumonia:
    test_image_pneumonia = os.path.join(testing_data_path, str(image))
    shutil.copy(test_image_pneumonia, 'C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/test/Pneumonia')

for image in testing_images_normal:
    test_image_normal = os.path.join(testing_data_path, str(image))
    shutil.copy(test_image_normal, 'C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/test/Normal')



batch_size = 32


train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,rotation_range=0.2,
                                   width_shift_range=0.2,height_shift_range=0.2,
                                   zoom_range=0.2,horizontal_flip=True,
                                   vertical_flip=True,fill_mode='nearest')
                                  
train_generator = train_datagen.flow_from_directory('C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/train',
                                                    target_size=(224,224),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory('C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/val',
                                                    target_size=(224,224),
                                                    batch_size=batch_size,
                                                    class_mode='binary')


base_model = InceptionV3(include_top=False, weights='imagenet', pooling='max', input_shape=(224,224,3))

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = Dropout(0.3)(x)
x = Dense(1024, activation = "relu")(x)
x = Dropout(0.3)(x)
x = Dense(512, activation = "relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation = "sigmoid")(x)

model = keras.Model(inputs,outputs)

base_model.trainable = False
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(train_generator,
          steps_per_epoch=train_generator.samples//batch_size,
          epochs = 10,
          validation_data=valid_generator,
          validation_steps=valid_generator.samples//batch_size)


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/test',
                                                    target_size=(224,224),
                                                    batch_size=batch_size,
                                                    class_mode='binary')
num_of_test_samples=624
model.evaluate(test_generator)
Y_pred = model.predict_generator(test_generator, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)


test_generator.classes

pred=model.predict(test_generator)
predict=np.argmax(pred,axis=1)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_generator.classes, predict))
print('Classification Report')
target_names = ['Pneumonia', 'Normal']
print(classification_report(test_generator.classes, predict, target_names=target_names))
#----------------------------------------------------------------------------------------
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adamax
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix

batch_size = 32


train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,rotation_range=0.2,
                                   width_shift_range=0.2,height_shift_range=0.2,
                                   zoom_range=0.2,horizontal_flip=True,
                                   vertical_flip=True,fill_mode='nearest')
                                  
train_generator = train_datagen.flow_from_directory('C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/train',
                                                    target_size=(150,150),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',shuffle=True,seed=1234,color_mode='rgb')

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory('C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/val',
                                                    target_size=(150,150),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',shuffle=True,seed=1234,color_mode='rgb')
inputs = tf.keras.layers.Input((150,150,3))
base_model=tf.keras.applications.xception.Xception(include_top=False, weights="imagenet",input_shape=(150,150,3), pooling='avg') 
x=base_model(inputs)
output=layers.Dense(2, activation='sigmoid')(x)
model=tf.keras.models.Model(inputs=inputs, outputs=output)
model.compile(Adamax(learning_rate=1e-4), loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(train_generator, validation_data=valid_generator, epochs=30)



fig, axes = plt.subplots(1, 2, figsize=(30, 10))
with plt.style.context(plt.style.available[8]):
    plt.suptitle("Accuracy and loss of train and validation set for each epoch.",fontsize=25)
    axes[0].plot(history.history["accuracy"],label="Train accuracy")
    axes[0].plot(history.history["val_accuracy"],label="Validation accuracy")
    axes[0].legend(fontsize=25)
    axes[0].tick_params(axis="both", labelsize=20)
    axes[0].set_xlabel("Epochs",fontsize=25)
    axes[0].set_ylabel("Accuracy",fontsize=25)
    
    axes[1].plot(history.history["loss"],label="Train loss")
    axes[1].plot(history.history["val_loss"],label="Validation loss")
    axes[1].legend(fontsize=25)
    axes[1].tick_params(axis="both", labelsize=20)
    axes[1].set_xlabel("Epochs",fontsize=25)
    axes[0].set_ylabel("Loss",fontsize=25)


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/working/test',
                                                  target_size=(150,150),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',shuffle=True,seed=1234,color_mode='rgb')


preds=model.predict(test_generator)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
gt=test_generator.classes
print("Results on test set:")
print(classification_report(gt,preds,target_names=["Normal","Covid"]))
print("ROC AUC score:   ",roc_auc_score(gt,preds))
print("F1 score:",f1_score(gt,preds))


conf_matrix = confusion_matrix(gt, preds)
sns.heatmap(conf_matrix,xticklabels = ["Normal","Covid"], yticklabels =["Normal","Covid"],annot=True,fmt='g')
plt.title('Confusion Matrix')
plt.show()






