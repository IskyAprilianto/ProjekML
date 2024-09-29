import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import files
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Define ImageDataGenerator with augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.4  # Splitting for validation
)

# Define ImageDataGenerator for validation data
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.4
)

# Create generators for training and validation
train_generator = train_datagen.flow_from_directory(
    'rockpaperscissors/rps-cv-images',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    'rockpaperscissors/rps-cv-images',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Avoid overfitting
    layers.Dense(256, activation='relu'),  # Additional hidden layer
    layers.Dropout(0.5),  # Avoid overfitting
    layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10  # Adjust based on your training time
)

# Function to upload an image
def upload_image():
    uploaded = files.upload()  # Opens a dialog to upload files
    for file_name in uploaded.keys():
        return file_name  # Return the name of the uploaded file

# Function to predict and display the image
def predict_and_show_image(img_path):
    # Load and display the image
    img = image.load_img(img_path, target_size=(150, 150))
    plt.imshow(img)
    plt.axis('off')  # Hide axis
    plt.show()

    # Prepare the image for prediction
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.

    # Make prediction
    result = model.predict(img_array)
    print(f"Hasil prediksi (softmax): {result}")  # Show softmax results

    classes = ['rock', 'paper', 'scissors']
    predicted_class = classes[np.argmax(result)]
    print(f"Gambar diprediksi sebagai: {predicted_class}")

# Call the upload function
image_path = upload_image()  # This will prompt you to upload an image

# Predict using the uploaded image
predict_and_show_image(image_path)
