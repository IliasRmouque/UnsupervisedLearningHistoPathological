import numpy as np

def process_data(data):
    raw_data = np.asarray(list(data.item().values()), dtype=np.float_)
    labels = list(data.item().keys())
    return raw_data, labels

def flatten_raw_data(raw_data):
    return raw_data.reshape((raw_data.shape[0], np.prod(raw_data.shape[1:])))