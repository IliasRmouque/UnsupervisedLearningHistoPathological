import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

"""
encoder_input = keras.Input(shape=(64, 64, 3), name="img")
x = layers.Conv2D(64, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, (3,3), activation="relu")(x)
x = layers.MaxPooling2D((3,3))(x)
x = layers.Conv2D(32, (3,3), activation="relu")(x)
x = layers.Conv2D(16, (3,3), activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, (3,3), activation="relu")(x)
x = layers.Conv2DTranspose(32, (3,3), activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, (3,3), activation="relu")(x)
decoder_output = layers.Conv2DTranspose(3, kernel_size=(39,39), activation="relu")(x)

autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()
data = np.load("patchesArray.npy")
data = data[100:]
"""

def build_autoencoder(img_shape):
    input_img = Input(shape=img_shape)
    x = MaxPooling2D((2, 2), padding='same')(input_img)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)
    return keras.Model(input_img, decoded)

def train_autoenc(autoencoder, data, epochs=10):

    #https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10) # permet d'éviter de surentrainer le modele
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True) # permet de sauvegarder le meilleur modele

    train_x, val_x, train_y, val_y = train_test_split(data, data, test_size=0.30, random_state=17)
    val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size=0.50, random_state=17)
    history = autoencoder.fit(train_x, train_x, epochs=epochs, validation_data=(val_x, val_x), callbacks=[es, mc])
    test_scores = autoencoder.evaluate(test_x, test_y, verbose=2)
    print("Test accuracy:", test_scores)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def evaluate_autoenc(data, autoencoder, nbr=10):
    alp = random.choices(data, k=nbr)

    al = autoencoder.predict(np.asarray(alp))
    fig = plt.figure()

    for i in range(nbr):
        ax = fig.add_subplot(2, nbr, i + 1)
        img = alp[i]
        plt.imshow(img)
        plt.axis("off")
        ax.set_title(i)
        imgEvaluate = al[i]
        ax = fig.add_subplot(2, nbr, nbr + i + 1)
        plt.imshow(imgEvaluate)
        ax.set_title(i)
        plt.axis("off")
    plt.show()

if __name__ == '__main__':
    data = np.load("patchesArray.npy")
    data = data[-1000:]/255
    data[:,:,:,1]= 0
    print(data[0])

    autoencoder = build_autoencoder(data.shape[1:])
    autoencoder.compile(optimizer='adam', loss='mean_squared_error',metrics=[tf.keras.metrics.Accuracy()] )
    autoencoder.summary()
    train_autoenc(autoencoder, data, 200)
    evaluate_autoenc(data, autoencoder)








