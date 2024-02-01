import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

# Load InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Add Custom Head for Bullseye Detection
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze Pre-trained Layers (Optional)
for layer in base_model.layers:
    layer.trainable = False

# Compile the Model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Specify the path to your extracted dataset
dataset_path = './2015-bullseye-1/'

# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32  # Adjust the batch size as needed

# Define data generators for training set
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary')

# Create training data generator
validation_generator = test_datagen.flow_from_directory(
    dataset_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary')

# Train the Model
steps_per_epoch = len(train_generator) // batch_size
validation_steps = len(validation_generator) // batch_size

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_steps)

# Save the Model
model.save('bullseye_detection_model.h5')
