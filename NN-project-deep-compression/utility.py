import os
import gzip
import shutil
import urllib
import numpy as np
import struct
from sklearn import datasets, model_selection
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

'''
Data Management Functions
'''


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def download_one_file(download_url,
                      local_dest,
                      expected_byte=None,
                      unzip_and_remove=False):
    """
    Download the file from download_url into local_dest
    if the file doesn't already exists.
    If expected_byte is provided, check if
    the downloaded file has the same number of bytes.
    If unzip_and_remove is True, unzip the file and remove the zip file
    """
    if os.path.exists(local_dest) or os.path.exists(local_dest[:-3]):
        print('%s already exists' % local_dest)
    else:
        print('Downloading %s' % download_url)
        local_file, _ = urllib.request.urlretrieve(download_url, local_dest)
        file_stat = os.stat(local_dest)
        if expected_byte:
            if file_stat.st_size == expected_byte:
                print('Successfully downloaded %s' % local_dest)
                if unzip_and_remove:
                    with gzip.open(local_dest, 'rb') as f_in, open(local_dest[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(local_dest)
            else:
                print('The downloaded file has unexpected number of bytes')


def download_mnist(path):
    """
    Download and unzip the dataset mnist if it's not already downloaded
    Download from http://yann.lecun.com/exdb/mnist
    """
    safe_mkdir(path)
    url = 'http://yann.lecun.com/exdb/mnist'
    filenames = ['train-images-idx3-ubyte.gz',
                 'train-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz',
                 't10k-labels-idx1-ubyte.gz']
    expected_bytes = [9912422, 28881, 1648877, 4542]

    for filename, byte in zip(filenames, expected_bytes):
        download_url = os.path.join(url, filename)
        local_dest = os.path.join(path, filename)
        download_one_file(download_url, local_dest, byte, True)


def parse_data(path, dataset, flatten):
    if dataset != 'train' and dataset != 't10k':
        raise NameError('dataset must be train or t10k')

    label_file = os.path.join(path, dataset + '-labels-idx1-ubyte')
    with open(label_file, 'rb') as file:
        _, num = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.int8)  # int8
        new_labels = np.zeros((num, 10))
        new_labels[np.arange(num), labels] = 1

    img_file = os.path.join(path, dataset + '-images-idx3-ubyte')
    with open(img_file, 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)  # uint8
        imgs = imgs.astype(np.float32) / 255.0
        if flatten:
            imgs = imgs.reshape([num, -1])

    return imgs, new_labels


def read_mnist(path, flatten=True, num_train=55000):
    """
    Read in the mnist dataset, given that the data is stored in path
    Return two tuples of numpy arrays
    ((train_imgs, train_labels), (test_imgs, test_labels))
    """
    imgs, labels = parse_data(path, 'train', flatten)
    indices = np.random.permutation(labels.shape[0])
    if (num_train == -1):
        num_train = labels.shape[0]
    train_idx, val_idx = indices[:num_train], indices[num_train:]

    train_img, train_labels = imgs[train_idx, :], labels[train_idx, :]
    val_img, val_labels = imgs[val_idx, :], labels[val_idx, :]
    test = parse_data(path, 't10k', flatten)

    return (train_img, train_labels), (val_img, val_labels), test


def get_iris_dataset():
    dataset = datasets.load_iris()
    X = dataset['data']
    y = dataset['target']
    Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, y, stratify=y)
    return Xtrain, Xtest, ytrain, ytest


'''
Pruning Functions
'''

'''
According to Song Hans previous paper (Learning both Weights and Connections for Efficient Neural Networks),
the pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer’s weights'

'''


def prune_weigth(original_weigth, threshold=0.25, std_smooth=True):
    if (std_smooth):
        threshold = np.std(original_weigth) * threshold

    super_threshold_indices = np.abs(original_weigth) < threshold
    original_weigth[super_threshold_indices] = 0
    return super_threshold_indices


'''
Quantization Functions
'''


def get_quantized_weight(layer_weight, bits=4):
    min_ = layer_weight.min()
    max_ = layer_weight.max()
    space = np.linspace(min_, max_, num=2 ** bits)
    kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1, precompute_distances=True,
                    algorithm="full")
    kmeans.fit(layer_weight.reshape(-1, 1))
    ris = kmeans.cluster_centers_[kmeans.labels_].reshape(layer_weight.shape)
    return ris,kmeans
