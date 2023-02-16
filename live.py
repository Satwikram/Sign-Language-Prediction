# 1) load the model
from tensorflow import keras
model = keras.models.load_model('trained-models/mobilenet/model.h5')


import cv2
from labels import *

import numpy as np


def normalize(img):

    img = cv2.resize(img, (224, 224))
    img = img / 255
    img = np.expand_dims(img, axis=0)
    return img


cap = cv2.VideoCapture(0)

#Check if camera was opened correctly
if not (cap.isOpened()):
    print("Could not open video device")

# 2) fetch one frame at a time from your camera
while(True):
    
    ret, frame = cap.read()
    
    input_image = normalize(frame)
    prediction = model.predict([input_image])
    prediction = np.argmax(prediction, -1)[0]

    prediction = labels[prediction]

    prediction_path = 'media/labels/' + prediction + '.jpg'

    prediction = "The Predicted Alphabet is: "+str(prediction)
    print(prediction)
    
    cv2.putText(frame, prediction, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)


    cv2.imshow("Prediction", frame)

    #Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()