from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom top layers for binary classification
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Specify the path to your extracted dataset
dataset_path = './2015-bullseye-1/'

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

# Train the model
epochs = 10  # You can adjust the number of epochs
model.fit(train_data_generator, epochs=epochs, validation_data=validation_data_generator)
