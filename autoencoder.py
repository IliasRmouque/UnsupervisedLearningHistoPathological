import csv
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2DTranspose,Lambda, Reshape, Flatten, Input, Dense, Conv2D,MaxPooling3D, MaxPooling2D, UpSampling2D, UpSampling3D, BatchNormalization, Activation
from tensorflow.keras import backend as K, Model, metrics, optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint, EarlyStopping


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


from utils.path import *


def train( raw_data, vae, encoder, decoder,  epochs = 100):
    """ train VAE model
    """

    train_datagen = ImageDataGenerator(rescale=1. / (LOD - 1),
                                       horizontal_flip=True,
                                       vertical_flip=True)

    # colormode needs to be set depending on num_channels

    print('using three channel generator!')
    # train_generator = train_datagen.flow_from_directory(
    #     IMG_DIR,
    #     target_size=(image_size, image_size),
    #     batch_size=16,
    #     color_mode='rgb',
    #     class_mode='input')


    # if files saved as single npy block

    # instantiate callbacks
    callbacks = []

    term_nan = TerminateOnNaN()
    callbacks.append(term_nan)

    save_dir='./VAE/sv'
    csv_logger = CSVLogger(os.path.join(save_dir, 'training.log'),
                           separator='\t')
    callbacks.append(csv_logger)

    checkpointer = ModelCheckpoint(os.path.join(save_dir, 'checkpoints/vae_weights.hdf5'),
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=True)
    callbacks.append(checkpointer)

    earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=20)
    callbacks.append(earlystop)

    model_dir = './VAE/models'
    history = vae.fit(raw_data,epochs=epochs,callbacks=callbacks)

    print('saving model weights to', model_dir)
    vae.save_weights(os.path.join(model_dir, 'weights_vae.hdf5'))
    encoder.save_weights(os.path.join(model_dir, 'weights_encoder.hdf5'))
    decoder.save_weights(os.path.join(model_dir, 'weights_decoder.hdf5'))

    print('done!')



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
    x = Dense(1, activation='relu', name='endEncoder')(x)
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
    return keras.Model(input_img, decoded, name="basic_with_dense")

def build_autoencoder_1(img_shape):
    input_img = Input(shape=img_shape)
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # =====
    x = Conv2D(32, (2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # =====
    sh = x.shape
    x = Flatten( name='endEncoder')(x)
    # encoder

    x = Reshape(sh[1:])(x)
    # =======
    x = Conv2D(32, (2, 2), padding='same')(x)
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
    return keras.Model(input_img, decoded, name="basic_no_dense")


def build_autoencoder2(img_shape):
    input_img = Input(shape=img_shape)
    # =====
    x = Conv2D(128, (3, 3), padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # =====
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # =====
    x = Conv2D(32, (2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # =====
    sh = x.shape
    x = Flatten()(x)
    # x = Dense(np.prod(sh[1:]), activation='relu', name='endEncoder')(x)
    # encoder

    x=  Reshape(sh[1:])(x)
    # =======
    x = Conv2D(32, (2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    # =======
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    # =======
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    # =======
    x = Conv2D(3, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)
    return keras.Model(input_img, decoded, name="basic_no_dense_128")

def build_autoencoder_3(img_shape, nlayers, zsize, finaldim):
    input_img = Input(shape=img_shape)
    x=input_img
    for i in range(nlayers):
        x = Conv2D(finaldim*2**(nlayers-i-1), (nlayers-i+1, nlayers-i+1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    shape = x.shape

    if zsize != 0:
        x = Flatten()(x)
        x = Dense(zsize, activation='relu', name='endEncoder')(x)

    encoder = Model(input_img, x, name='encoder')

    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    for i in range(nlayers):
        x = Conv2D(finaldim*2**(i), ( 2+i, 2+i), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)
    return keras.Model(input_img, decoded, name="basic_dense_128"), encoder





def train_autoenc(autoencoder, data, epochs=10, patience=10, plot =True, fName='best_model.h5'):

    # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience) # permet d'éviter de surentrainer le modele
    mc = ModelCheckpoint(fName, monitor='val_loss', mode='min', verbose=1, save_best_only=True) # permet de sauvegarder le meilleur modele

    # dividing the dataset
    train_x, val_x, train_y, val_y = train_test_split(data, data, test_size=0.30, random_state=17)
    val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size=0.50, random_state=17)

    # training
    history = autoencoder.fit(train_x, train_x,  epochs=epochs, validation_data=(val_x, val_x), batch_size=100 ,callbacks=[es, mc])
    if 'test' not in history.history:
        history.history['test'] =[]
    test_scores = autoencoder.evaluate(test_x, test_y, verbose=2)
    history.history['test'].append(test_scores)
    print("Test accuracy:", test_scores)
    if plot:
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
    return history


def evaluate_network(data, network, nbr=10):
    original = random.sample(data, k=nbr)
    evaluated = network.predict(np.asarray(original))
    return original, evaluated


def show_evaluation(original,evaluated, title=None):
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
    if title:
        plt.savefig(title + ".png", format="png")
    else:
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
    i=1
    max=0
    for layer in autoencoder.layers[1:]:
        i+=1
        encoder_model = layer(encoder_model)
        if layer.name == 'endEncoder':
            encoder_model = keras.Model(inputs=encoder_input, outputs=encoder_model)
            return encoder_model
        if "pool" in layer.name:
            max=i
    encoder_model = encoder_input
    for layer in autoencoder.layers[1:max]:
        encoder_model = layer(encoder_model)
    encoder_model = keras.Model(inputs=encoder_input, outputs=encoder_model)
    return encoder_model
def train_with_big_ds(autoencoder, data, batchsize, namedir):
    nbBatch = len(list(data.values())) // batchsize + 1  # number of batch of size 100000 (easily predictable)
    print("there will be {} trainings".format(nbBatch))
    history = False
    for i in range(0, nbBatch):
        raw_data = np.asarray(list(data.values())[i * batchsize: (i + 1) * batchsize]) / 255
        if not history:
            history = train_autoenc(autoencoder, data=raw_data, epochs=100, patience=50, plot=False,
                                    fName=namedir + 'best_model.h5')
        else:
            orig, ev = evaluate_network(list(raw_data), autoencoder)
            show_evaluation(orig, ev, title=namedir + "sub_batch:" + str(i))
            hist_temp = train_autoenc(autoencoder, data=raw_data, epochs=100, patience=20, plot=False,
                                      fName=namedir + 'best_model.h5')
            for k in hist_temp.history.keys():
                history.history[k] += hist_temp.history[k]

        plt.clf()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.savefig(namedir + "history.png")
    with open("./ploud.csv", 'a') as f:
        a = [namedir]
        a.append(history.history['loss'][-1])
        a.append(history.history['val_loss'][-1])
        a.append(history.history['test'][-1])
        rcsv = csv.writer(f)
        rcsv.writerow(a)

def predict_with_big_ds(encoder, raw_data, labels, batchsize):
    res = {}
    nbBatch = len(list(data.values())) // batchsize + 1  # number of batch of size 100000 (easily predictable)
    for idx in range(nbBatch):
        print(idx, '/', nbBatch)
        array_of_batches = np.asarray(raw_data[idx * batchsize: (idx + 1) * batchsize])
        predicted = encoder.predict(array_of_batches / 255)
        for i in range(len(predicted)):
            res[labels[i + idx * batchsize]] = predicted[i]
    res = dict(zip(labels, flatten_raw_data(np.asarray(list(res.values())))))
    return res

def split_autencoder_decoder(autoencoder):
    encoder_input = Input(autoencoder.layers[10].input_shape[1:])
    encoder_model = encoder_input
    for layer in autoencoder.layers[11:]:
        encoder_model = layer(encoder_model)
    encoder_model = keras.Model(inputs=encoder_input, outputs=encoder_model)
    return encoder_model


if __name__ == '__main__':
#     data = (np.load(IMG_DIR + "../patchesArray.npy", allow_pickle=True))
#     raw_data = np.asarray(list(data.item().values())) / 255
#     print(raw_data.shape)
#     vae, encoder,decoder = build_model_VAE(raw_data.shape[1:], 3)
#     train(raw_data, vae, encoder, decoder, epochs=100)
#
#
#
    final=[]
    from main import *
    nb_clust = 7
    dct_patch = {}
    with open(IMG_DIR + "../tile_selection2.tsv") as f:
        rcsv = csv.reader(f, delimiter="\t")
        # read the first line that holds column labels
        csv_labels = rcsv.__next__()
        for record in rcsv:
            if record[3] == '1':
                dct_patch[record[0]] = Patch(record[0], size=ptch_size, row=int(record[4]),
                                             column=int(record[5]))
    mask_type = load_annotations("./annoWeird.csv", "./anno.csv", IMG_ID, nb_clust)
    # loading data
    for LOD in [4]:# 2, 4, 8, 16]:
        IMG_DIR = './../PyHIST/output/' + IMG_NAME + '_' + str(ptch_size) + '*' + str(ptch_size) \
              + "_LOD=" + str(LOD) + '/' + IMG_NAME + '_tiles/'
        data = (np.load(IMG_DIR + "../patchesArray.npy", allow_pickle=True))
        raw_data, labels = process_data(data)
        raw_data = raw_data
        data = dict(zip(labels, raw_data))
        data1 = dict(list(data.items())[len(data) // 2:])
        data2 = dict(list(data.items())[:len(data) // 2])
        rd=np.asarray(list(data.values()))
        final={}
        ploud=0
        for data in [data1, data2]:
            print(IMG_DIR)
            for siz in [2]:
               for zsiz in [32]:#128,256,512,1024]:
                   for dim_mid in [32]:
                     #try:

                        raw_data = np.asarray(list(data.values()))
                        #labels = list(data.keys())
                        raw_data = raw_data / 255
                        print(raw_data[0])
                        namedir="./Autoenco5/test_ae_nblayer=" +str(siz) + "_zsize=" + str(zsiz) + "dim_mid=" + str(dim_mid) +'/'
                        os.makedirs(namedir, exist_ok=True)
                        # # building the autoencoder
                        #autoencoder = build_autoencoder(raw_data.shape[1:])
                        autoencoder, enco = build_autoencoder_3(raw_data.shape[1:], nlayers=siz, zsize=zsiz, finaldim=dim_mid)
                        #autoencoder.summary()
                        autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
                        # autoencoder.summary()
                        #train_autoenc(autoencoder, data=raw_data, epochs=20000, patience=100)

                        #getting the encoder
                        #autoencoder = keras.models.load_model(namedir + './best_model.h5')
                        keras.utils.plot_model( autoencoder, to_file=namedir +"autoenco.png", show_shapes=True, show_layer_names=True)
                        batchsize = 1000
                        train_with_big_ds(autoencoder, data, batchsize, namedir)

                        res = predict_with_big_ds(enco, rd, labels, batchsize)
                        np.save(IMG_DIR + '../pa'+str(ploud), res)
                        # plt.clf()
                        # ax = plt.subplot(111)
                        # print(len(res))
                        ploud+=1
                        for k, i in res.items():
                            final[k]=i
                        # get_UMAP(res, ax, nb_clust, get_termlist(dct_patch, mask_type, 0, IMG_SIZE, LOD), nb_limit=500)
                        # plt.savefig(namedir + "UMAP.png", format="png")
                     # except Exception as e:
                     #     print(e)
                     #     final.append(namedir)
                     #     pass


            #os.rename('best_model_'+str(i)+ '.h5', 'best_model_'+ autoencoder.name + "_" + str(LOD)+ str(ptch_size) + '.h5')

                    # enco = split_autencoder(autoencoder)
                    # # keras.utils.plot_model(enco, "enco.png")
                    # enco.summary()

        np.save(IMG_DIR + '../double_predicted', final)
                # visual evaluation
                # orig, ev = evaluate_network(list(raw_data), autoencoder)
                # show_evaluation(orig, ev)



                # predict: with big data_


                # np.save(IMG_DIR + '../predicted_array3', res)









