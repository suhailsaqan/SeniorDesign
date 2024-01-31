from tensorflow.keras.preprocessing.image import ImageDataGenerator
from roboflow import Roboflow

rf = Roboflow(api_key="your_api_key")  # Replace "your_api_key" with your actual Roboflow API key
project = rf.workspace("paingkl").project("2015-bullseye")
dataset = project.version(1).download("tensorflow")

# The downloaded dataset is typically in a zip file, so you need to extract it
import zipfile
import os

with zipfile.ZipFile(dataset, 'r') as zip_ref:
    zip_ref.extractall('./')  # Extract to the current directory

# Specify the path to your extracted dataset
dataset_path = './export/'

# Define data generators for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Assuming you want to use 80% for training and 20% for validation
)

# Create training data generator
train_data_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # Adjust the target size based on your model requirements
    batch_size=32,
    class_mode='binary',  # 'binary' for binary classification
    subset='training'
)

# Create validation data generator
validation_data_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# The train_data variable now contains a generator that yields batches of training data during training
# Likewise, validation_data contains batches for validation
