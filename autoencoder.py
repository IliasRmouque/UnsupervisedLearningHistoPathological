import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Reshape, Flatten, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
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
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dense(16, activation='relu', name='endEncoder')(x)
    # encoder

    x = Conv2D(16, (3, 3), padding='same')(x)
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

def build_autoencoder2(img_shape):
    input_img = Input(shape=img_shape)
    # =====
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # =====
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # =====
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # =====
    sh = x.shape
    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='endEncoder')(x)
    # encoder

    x=  Reshape(sh[1:])(x)
    # =======
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    # =======
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    # =======
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    # =======
    x = Conv2D(3, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)
    return keras.Model(input_img, decoded)

def train_autoenc(autoencoder, data, epochs=10, patience=10):

    # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience) # permet d'éviter de surentrainer le modele
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True) # permet de sauvegarder le meilleur modele

    # dividing the dataset
    train_x, val_x, train_y, val_y = train_test_split(data, data, test_size=0.30, random_state=17)
    val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size=0.50, random_state=17)

    # training
    history = autoencoder.fit(train_x, train_x, epochs=epochs, validation_data=(val_x, val_x), callbacks=[es, mc])
    test_scores = autoencoder.evaluate(test_x, test_y, verbose=2)
    print("Test accuracy:", test_scores)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def evaluate_network(data, network, nbr=10):
    original = random.choices(data, k=nbr)
    evaluated = network.predict(np.asarray(original))
    return original, evaluated


def show_evaluation(original,evaluated):
    nbr = len(evaluated)
    fig = plt.figure()
    for i in range(nbr):
        ax = fig.add_subplot(2, nbr, i + 1)
        img = original[i]
        plt.imshow(img)
        plt.axis("off")
        ax.set_title(i)
        imgEvaluate = evaluated[i]
        ax = fig.add_subplot(2, nbr, nbr + i + 1)
        plt.imshow(imgEvaluate)
        ax.set_title(i)
        plt.axis("off")
    plt.show()

def full_processing(raw_data, encoder, labels):
    a= encoder.predict([raw_data])
    res={}
    for  i, pred in enumerate(a):
        res[labels[i]]=pred
    np.save('./autoencoded_pictures', res)
    return a


def split_autencoder(autoencoder):
    encoder_input = Input(autoencoder.layers[1].input_shape[1:])
    encoder_model = encoder_input
    for layer in autoencoder.layers[1:11]:
        encoder_model = layer(encoder_model)
        if layer.name == 'endEncoder':
            break
    encoder_model = keras.Model(inputs=encoder_input, outputs=encoder_model)
    return encoder_model

def split_autencoder_decoder(autoencoder):
    encoder_input = Input(autoencoder.layers[10].input_shape[1:])
    encoder_model = encoder_input
    for layer in autoencoder.layers[11:]:
        encoder_model = layer(encoder_model)
    encoder_model = keras.Model(inputs=encoder_input, outputs=encoder_model)
    return encoder_model


if __name__ == '__main__':
    #loading data
    data = (np.load("../Images/Test2/patchesArray.npy", allow_pickle=True))
    raw_data= np.asarray(list(data.item().values())[:10000])/255
    labels = np.asarray(list(data.item().keys())[:10000])


    # building the autoencoder
    autoencoder = build_autoencoder2(raw_data.shape[1:])
    autoencoder.summary()
    autoencoder.compile(optimizer='adadelta', loss=tf.keras.losses.MeanSquaredError())
    train_autoenc(autoencoder, data=raw_data, epochs=2000, patience=10)

    #getting the encoder
    #autoencoder = keras.models.load_model('best_model.h5')
    keras.utils.plot_model(autoencoder, "autoenco.png")
    #autoencoder.summary()

    enco = split_autencoder(autoencoder)
    keras.utils.plot_model(enco, "enco.png")
    #enco.summary()


    #visual evaluation
    orig, ev = evaluate_network(raw_data, autoencoder)
    show_evaluation(orig, ev)


    #predict: normal way
    #raw_data= np.asarray(list(data.item().values()))/255
    #full_processing(raw_data, enco, labels)

    #predict: with big data_
    batchsize=10000# batch of size 100000 (easily predictable)
    print(len(raw_data))
    nbBatch = len(raw_data)//batchsize +1 #number of batch of size 100000 (easily predictable)
    res={}
    for mara in range(nbBatch):
        print(mara, '/', nbBatch)
        array_of_batches=np.asarray(raw_data[mara*batchsize : (mara+1)*batchsize] )
        predicted = enco.predict(array_of_batches/255)
        for i in range(len(predicted)):
            res[labels[i+mara*batchsize]]= predicted[i]




    np.save('./madness', res)








