import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Load your trained model
model = load_model('bullseye_detection_model.h5')  # Replace with the actual path to your model

# Open a connection to the camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the frame to match the input size of the model (299x299 for InceptionV3)
    resized_frame = cv2.resize(frame, (299, 299))

    # Preprocess the frame for prediction
    img_array = image.img_to_array(resized_frame)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make a prediction
    prediction = model.predict(img_array)

    print(prediction[0][0])

    # Extract the prediction result
    if prediction[0][0] > 0.5:  # Adjust the threshold as needed
        label = 'Bullseye'
    else:
        label = 'Not Bullseye'

    # Display the result on the frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Bullseye Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
