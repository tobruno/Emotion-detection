import pandas as pd
import numpy as np
import cv2
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import to_categorical

from keras.models import model_from_json
from keras.utils import img_to_array
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt




df = pd.read_csv('fer2013.csv')
emotions = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprised", 6:"Neutral"}

x_train, x_test, x_val = [], [], []
y_train, y_test, y_val = [], [], []
for index, row in df.iterrows():
    k = row['pixels'].split(" ")
    if row['Usage'] == 'Training':
        x_train.append(np.array(k))
        y_train.append(row['emotion'])
    elif row['Usage'] == 'PrivateTest':
        x_test.append(np.array(k))
        y_test.append(row['emotion'])
    elif row['Usage'] == 'PublicTest':
        x_val.append(np.array(k))
        y_val.append(row['emotion'])

x_train = np.array(x_train, dtype='float')
x_test = np.array(x_test, dtype='float')
x_val = np.array(x_val, dtype='float')
y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)

#Zmiana formatu dla zdjec
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_val = x_val.reshape(x_val.shape[0], 48, 48, 1)
y_train = to_categorical(y_train, num_classes = 7)
y_test = to_categorical(y_test, num_classes = 7)
y_val = to_categorical(y_val, num_classes = 7)


def Model():
    model = Sequential()
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), activation='relu', padding="same", input_shape=((48, 48, 1))))
    model.add(Convolution2D(32, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Convolution2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Convolution2D(96, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Convolution2D(128, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(7, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, epsilon=0.0001, patience=1, verbose=1)
    history = model.fit(x_train, y_train, batch_size=128, callbacks=lr_reduce, validation_data=(x_train, y_train), epochs=50)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    #return model


def detection():
    model = model_from_json(open("model.json", "r").read())
    image = cv2.imread('C:/Users/user/Documents/GitHub/Projects/Emotion-detection/webApp/instance/photo/image.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (48, 48))
    image_pixels = img_to_array(gray_image)  # converting image to array
    image_pixels = np.expand_dims(image_pixels, axis=0)

    predictions = model.predict(image_pixels)  # model prediction
    max_index = np.argmax(predictions[0])  # getting emotion index

    emotion_detection = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral')
    emotion_prediction = emotion_detection[max_index]
    print(emotion_prediction)


if __name__ == '__main__':

    #Model()
    detection()


