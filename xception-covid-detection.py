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
from plot_keras_history import show_history, plot_history
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df_meta=pd.read_csv("C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/Chest_xray_Corona_Metadata.csv",index_col=0)
df_meta.head()

train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input,zoom_range=0.1,brightness_range=[0.5,1.3],
                                   width_shift_range=0.1,height_shift_range=0.1,validation_split=0.1)
test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input)


BATCH_SIZE=64
path="C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"
train_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='training', directory=path+"/train")

val_images=train_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TRAIN"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=True,seed=1234,subset='validation', directory=path+"/train")

test_images = test_datagen.flow_from_dataframe(dataframe=df_meta[df_meta["Dataset_type"]=="TEST"],x_col='X_ray_image_name',y_col='Label',color_mode='rgb',class_mode='categorical',
                                                target_size=(150,150),batch_size=BATCH_SIZE,shuffle=False, directory=path+"/test")


inputs = tf.keras.layers.Input((150,150,3))
base_model=tf.keras.applications.xception.Xception(include_top=False, weights="imagenet",input_shape=(150,150,3), pooling='avg') 
x=base_model(inputs)
output=layers.Dense(2, activation='sigmoid')(x)
model=tf.keras.models.Model(inputs=inputs, outputs=output)
model.compile(Adamax(learning_rate=1e-4), loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(train_images, validation_data=val_images, epochs=30)


show_history(history)
plot_history(history, path="C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/Training_history.png",
             title="Training history")
plt.close()
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
preds=preds>0.5
gt=[0 if x=="Normal" else 1 for x in df_meta[df_meta["Dataset_type"]=="TEST"]["Label"]]
print("Results on test set:")
print(classification_report(gt,preds,target_names=["Normal","Covid"]))
print("ROC AUC score:   ",roc_auc_score(gt,preds))
print("F1 score:",f1_score(gt,preds))
conf_matrix = confusion_matrix(gt, preds)
sns.heatmap(conf_matrix,xticklabels = ["Normal","Covid"], yticklabels =["Normal","Covid"],annot=True,fmt='g')
plt.title('Confusion Matrix')
plt.show()
plt.savefig("C:/Users/GIGABYTE/Downloads/CoronaHack _Chest X-Ray_classify/Confusion Matrix.png")

