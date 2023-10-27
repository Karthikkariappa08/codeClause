# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
# Load dataset
train_df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')
test_df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/test.csv')

# Add '.png' extension to id_code
train_df['id_code'] = train_df['id_code'].apply(lambda x: x + ".png")
test_df['id_code'] = test_df['id_code'].apply(lambda x: x + ".png")

# Convert the labels to string
train_df['diagnosis'] = train_df['diagnosis'].astype('str')
# normalize the images
train_df['diagnosis'] = train_df['diagnosis'].astype('str')

# Split the data
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Generate batches of tensor image data for train and validation
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='/kaggle/input/aptos2019-blindness-detection/train_images/',
    x_col='id_code',
    y_col='diagnosis',
    batch_size=32,
    class_mode='categorical',
    target_size=(224, 224))

val_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory='/kaggle/input/aptos2019-blindness-detection/train_images/',
    x_col='id_code',
    y_col='diagnosis',
    batch_size=32,
    class_mode='categorical',
    target_size=(224, 224))
# Get the ResNet50 base model
base_model = ResNet50(include_top=False, pooling='avg')

# Make the base model untrainable
base_model.trainable = False

# Define the model
model = Sequential([
    base_model,
    Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Define the callbacks
callbacks = [EarlyStopping(patience=3, restore_best_weights=True),
             ModelCheckpoint(filepath='model.h5', save_best_only=True)]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.n//train_generator.batch_size,
    validation_data = val_generator, 
    validation_steps = val_generator.n//val_generator.batch_size,
    epochs = 10,
    callbacks=callbacks)
loss, accuracy = model.evaluate(val_generator)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.show()
