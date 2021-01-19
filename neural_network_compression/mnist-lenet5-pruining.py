import tensorflow.python.eager as tfe
import tensorflow as tf
import numpy as np
import neural_network_compression.utility as utility
import matplotlib.pyplot as plt

def reset_seed():
    np.random.seed(0)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.random.set_seed(0)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


reset_seed()
mnist_folder = "data/mnist"

input_width = 28
input_height = 28
input_size = input_height * input_width
filter1_size = 20
filter2_size = 50
dense_size = 256
output_size = 10


class LeNet5(tf.keras.Model):
    """LeNet5 model"""

    def __init__(self):
        super(LeNet5, self).__init__()

        # weight: (5,5,1,20)=500 + (20,)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filter1_size, kernel_size=[5, 5], padding="same", activation=tf.nn.relu
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)

        # weight: (5,5,20,50)=25000 + (50,)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filter2_size, kernel_size=[5, 5], padding="same", activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)

        #  weight: (2450, 256)= 627200 + (256,)
        self.dense = tf.keras.layers.Dense(dense_size, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(0.2)

        # weight: (256, 10) = 2560 + (10,)
        self.logits = tf.keras.layers.Dense(units=output_size)

    def call(self, x, training=True):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        feature_dim = x.shape[1] * x.shape[2] * x.shape[3]
        x = tf.reshape(x, (-1, feature_dim))
        x = self.dense(x)
        x = self.dropout(x, training=training)
        return self.logits(x)

LENET5 = LeNet5()

def loss_func( x, y):

    """Loss function : softmax-cross-entropy with L2 regularization.

    Parameters
    ----------
    net : LeNet5
        The nerual network model.
    x : Tensor
        Value to predict the output.
    y: Tensor
        Correct prediction of x

    Returns
    -------
    double
        the average error
    """

    ypred = LENET5(x)
    beta = 0.01
    w1 = LENET5.conv1.get_weights()[0]
    w2 = LENET5.conv2.get_weights()[0]
    w3 = LENET5.dense.get_weights()[0]
    w4 = LENET5.logits.get_weights()[0]

    # not sure if it is correct
    l = tf.reduce_mean(tf.losses.BinaryCrossentropy()(y, ypred))

    #l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=ypred))
    regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4)
    loss_fin = tf.reduce_mean(l + beta * regularizers)
    return loss_fin


def get_grad(x,y):
    with tf.GradientTape() as tape:
        loss = loss_func(x, y)
    return tape.gradient(loss, LENET5.trainable_variables)

def apply_grad(opt,grads):
    opt.apply_gradients(zip(grads,LENET5.trainable_variables))

def print_zero_stat(net):

    """Prints the number of zeros in the neural network

    Parameters
    ----------
    net : LeNet300
        The nerual network model.

    Returns
    -------
    void
        print the number of zeros in each layer of the input neural network
        with respect the total weights in that layer
    """

    [l1, l1_b] = LENET5.conv1.get_weights()
    [l2, l2_b] = LENET5.conv2.get_weights()
    [d, d_b] = LENET5.dense.get_weights()
    [l, l_b] = LENET5.logits.get_weights()

    print(
        "Conv1 zeros:",
        np.count_nonzero(l1 == 0),
        " on ",
        l1.shape[0] * l1.shape[1] * l1.shape[2] * l1.shape[3],
    )
    print("Conv1 bias zeros :", np.count_nonzero(l1_b == 0), " on ", l1_b.shape[0])
    print(
        "Conv2 zeros :",
        np.count_nonzero(l2 == 0),
        " on ",
        l2.shape[0] * l2.shape[1] * l2.shape[2] * l2.shape[3],
    )
    print("Conv2 bias zeros :", np.count_nonzero(l2_b == 0), " on ", l2_b.shape[0])
    print("Dense zeros :", np.count_nonzero(d == 0), " on ", d.shape[0] * d.shape[1])
    print("Dense bias zeros :", np.count_nonzero(d_b == 0), " on ", d_b.shape[0])
    print("Logits zeros :", np.count_nonzero(l == 0), " on ", l.shape[0] * l.shape[1])
    print("Logits bias zeros :", np.count_nonzero(l_b == 0), " on ", l_b.shape[0])


def train_with_pruning(thresholds, std_smooth, normal_train, prune_train, semi_prune_train):
    """It sequentially  performs the normal train, prune train, semi-prune->( normal train without the pruned weights ).


    Parameters
    ----------
    thresholds : list
        list of the thresholds to use for every layer of the neural network.
    std_smooth : boolean
        if True each threshold is multiplied by the standard deviation of the respective layer of the neural network.
    normal_train : integer
        number of epochs to normally train the neural network
    prune_train : integer
        number of epochs to iteratively prune the neural network
    semi_prune_train : integer
        number of epochs to semi-prune the neural network (normal train without the pruned weights)

    How it works
    -------

    Download (only the first time) and splits the MNIST dataset,
    instantiate the LeNet300 neural network and its parameters.

    If the normal_train parameter is zero, it is assumed that the model has been previously normally trained
    and stored in the folder /checkpoint-lenet300-before-pruning

    The training starts and at the end of each epoch, it print the actual accuracy and the number of zeros in the nerual network

    After the last epoch of the normal_train phase, the model is stored in the folder /checkpoint-lenet300-before-pruning

    After the last epoch of the normal_train+prune_train+semi_prune phase,
    the model is stored in the folder /checkpoint-lenet300-after-pruning

    The logic of the prune_train phase is the following :
        1) get the matrix of weights of each layers of the neural network
        2) prune singularly each layer from the bias (side effect of the matrix)
            and get the indexes of the pruned matrix with the pruned weights
        3) set the new pruned weights in the neural network
        4) compute and apply the gradient : the pruned weights now will probably change the value so they will not be zero anymore
            but I have stored in step 2) the indexes of the pruned weights
        5) use the indexes of the pruned weights stored in step 2) to set to zero the current weights of the model

    The logic of the semi_prune phase is the following :
        1) use the indexes of the pruned weights previously stored during the last phase of the prune_train
        2) compute-apply gradients and set to zero the pruned weights using the indexes stored instep 1)


    Returns
    -------

    LeNet5
        the trained-pruned model
    list
        the accuracy values during the train

    """

    total_train_epoch = normal_train + prune_train + semi_prune_train
    accuracy_values = []

    utility.download_mnist(mnist_folder)
    [(xtrain, ytrain), (xval, yval), (xtest, ytest)] = utility.read_mnist(
        mnist_folder, flatten=True, num_train=-1
    )
    opt = tf.keras.optimizers.Adam(0.001)

    train_data = (
        tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        .shuffle(1000)
        .batch(512, drop_remainder=True)
    )

    xtest = tf.reshape(xtest, shape=[-1, input_width, input_height, 1])
    ytest = tf.argmax(ytest, axis=1)

    if normal_train == 0:

        root = tf.train.Checkpoint(
            optimizer=opt, model=LENET5, optimizer_step=tf.compat.v1.train.get_or_create_global_step()
        )
        root.restore(tf.train.latest_checkpoint("checkpoint-lenet5-before-pruning"))
        ypred = tf.nn.softmax(LENET5(tf.constant(xtest)))
        ypred = tf.argmax(ypred, axis=1)

        tmp = tf.cast(tf.equal(ypred, ytest), tf.float32)
        acc = tf.reduce_mean(tmp).numpy()
        print("\naccuracy before pruning : ", acc)

    for i in range(total_train_epoch):

        # semi prune : the previous zeros are left but no zeros will be added
        total_loss = 0
        if i >= normal_train + prune_train:

            for Xtrain_t, ytrain_t in train_data:
                Xtrain_t = tf.reshape(Xtrain_t, shape=[-1, input_width, input_height, 1])
                grads = get_grad(Xtrain_t, ytrain_t)
                #total_loss += current_loss.numpy()
                apply_grad(opt,grads)

                w1 = LENET5.conv1.get_weights()[0]
                w2 = LENET5.conv1.get_weights()[1]

                w3 = LENET5.conv2.get_weights()[0]
                w4 = LENET5.conv2.get_weights()[1]

                w5 = LENET5.dense.get_weights()[0]
                w6 = LENET5.dense.get_weights()[1]

                w7 = LENET5.logits.get_weights()[0]
                w8 = LENET5.logits.get_weights()[1]

                w1[zero_idx1] = 0
                w2[zero_idx2] = 0
                w3[zero_idx3] = 0
                w4[zero_idx4] = 0
                w5[zero_idx5] = 0
                w6[zero_idx6] = 0
                w7[zero_idx7] = 0
                w8[zero_idx8] = 0

                LENET5.conv1.set_weights([w1, w2])
                LENET5.conv2.set_weights([w3, w4])
                LENET5.dense.set_weights([w5, w6])
                LENET5.logits.set_weights([w7, w8])

        # prune
        if i >= normal_train and i < normal_train + prune_train:
            # prune only conv layer and block zero weigth
            if i <= normal_train + (prune_train) / 2:
                for Xtrain_t, ytrain_t in train_data:
                    Xtrain_t = tf.reshape(Xtrain_t, shape=[-1, input_width, input_height, 1])

                    # 1) get weight
                    w1 = LENET5.conv1.get_weights()[0]
                    w2 = LENET5.conv1.get_weights()[1]

                    w3 = LENET5.conv2.get_weights()[0]
                    w4 = LENET5.conv2.get_weights()[1]

                    w5 = LENET5.dense.get_weights()[0]
                    w6 = LENET5.dense.get_weights()[1]

                    w7 = LENET5.logits.get_weights()[0]
                    w8 = LENET5.logits.get_weights()[1]

                    # 2) side effect on the weights under threshold values and get pruned indexes

                    zero_idx1 = utility.prune_weigth(
                        w1, std_smooth=std_smooth, threshold=thresholds[0]
                    )
                    zero_idx2 = utility.prune_weigth(
                        w2, std_smooth=std_smooth, threshold=thresholds[1]
                    )
                    zero_idx3 = utility.prune_weigth(
                        w3, std_smooth=std_smooth, threshold=thresholds[2]
                    )
                    zero_idx4 = utility.prune_weigth(
                        w4, std_smooth=std_smooth, threshold=thresholds[3]
                    )

                    # block the zero value weight for the dense layers
                    zero_idx5 = w5 == 0
                    zero_idx6 = w6 == 0
                    zero_idx7 = w7 == 0
                    zero_idx8 = w8 == 0

                    # 3) set the neural network pruned weights
                    LENET5.conv1.set_weights([w1, w2])
                    LENET5.conv2.set_weights([w3, w4])
                    # LENET5.dense.set_weights([w5, w6])
                    # LENET5.logits.set_weights([w7, w8])

                    # 4) compute and apply the gradient optimization on the pruned neural network
                    grads = get_grad(Xtrain_t, ytrain_t)
                    #total_loss += current_loss.numpy()

                    apply_grad(opt,grads)

                    # 5) get the updated weights and zeroed the previously pruned weights
                    w1 = LENET5.conv1.get_weights()[0]
                    w2 = LENET5.conv1.get_weights()[1]

                    w3 = LENET5.conv2.get_weights()[0]
                    w4 = LENET5.conv2.get_weights()[1]

                    w5 = LENET5.dense.get_weights()[0]
                    w6 = LENET5.dense.get_weights()[1]

                    w7 = LENET5.logits.get_weights()[0]
                    w8 = LENET5.logits.get_weights()[1]

                    w1[zero_idx1] = 0
                    w2[zero_idx2] = 0
                    w3[zero_idx3] = 0
                    w4[zero_idx4] = 0
                    w5[zero_idx5] = 0
                    w6[zero_idx6] = 0
                    w7[zero_idx7] = 0
                    w8[zero_idx8] = 0

                    LENET5.conv1.set_weights([w1, w2])
                    LENET5.conv2.set_weights([w3, w4])
                    LENET5.dense.set_weights([w5, w6])
                    LENET5.logits.set_weights([w7, w8])

            # prune only dense layer and block zero weigth
            else:

                for Xtrain_t, ytrain_t in train_data:
                    Xtrain_t = tf.reshape(Xtrain_t, shape=[-1, input_width, input_height, 1])

                    # 1) get weight
                    w1 = LENET5.conv1.get_weights()[0]
                    w2 = LENET5.conv1.get_weights()[1]

                    w3 = LENET5.conv2.get_weights()[0]
                    w4 = LENET5.conv2.get_weights()[1]

                    w5 = LENET5.dense.get_weights()[0]
                    w6 = LENET5.dense.get_weights()[1]

                    w7 = LENET5.logits.get_weights()[0]
                    w8 = LENET5.logits.get_weights()[1]

                    # 2) side effect on the weights under threshold values and get pruned indexes

                    zero_idx1 = w1 == 0
                    zero_idx2 = w2 == 0
                    zero_idx3 = w3 == 0
                    zero_idx4 = w4 == 0
                    zero_idx5 = utility.prune_weigth(
                        w5, std_smooth=std_smooth, threshold=thresholds[4]
                    )
                    zero_idx6 = utility.prune_weigth(
                        w6, std_smooth=std_smooth, threshold=thresholds[5]
                    )
                    zero_idx7 = utility.prune_weigth(
                        w7, std_smooth=std_smooth, threshold=thresholds[6]
                    )
                    zero_idx8 = utility.prune_weigth(
                        w8, std_smooth=std_smooth, threshold=thresholds[7]
                    )

                    # 3) set the neural network pruned weights
                    # LENET5.conv1.set_weights([w1, w2])
                    # LENET5.conv2.set_weights([w3, w4])
                    LENET5.dense.set_weights([w5, w6])
                    LENET5.logits.set_weights([w7, w8])

                    # 4) compute and apply the gradient optimization on the pruned neural network
                    grads = get_grad(Xtrain_t, ytrain_t)

                    #total_loss += current_loss.numpy()

                    apply_grad(opt,grads)

                    # 5) get the updated weights and zeroed the previously pruned weights
                    w1 = LENET5.conv1.get_weights()[0]
                    w2 = LENET5.conv1.get_weights()[1]

                    w3 = LENET5.conv2.get_weights()[0]
                    w4 = LENET5.conv2.get_weights()[1]

                    w5 = LENET5.dense.get_weights()[0]
                    w6 = LENET5.dense.get_weights()[1]

                    w7 = LENET5.logits.get_weights()[0]
                    w8 = LENET5.logits.get_weights()[1]

                    w1[zero_idx1] = 0
                    w2[zero_idx2] = 0
                    w3[zero_idx3] = 0
                    w4[zero_idx4] = 0
                    w5[zero_idx5] = 0
                    w6[zero_idx6] = 0
                    w7[zero_idx7] = 0
                    w8[zero_idx8] = 0

                    LENET5.conv1.set_weights([w1, w2])
                    LENET5.conv2.set_weights([w3, w4])
                    LENET5.dense.set_weights([w5, w6])
                    LENET5.logits.set_weights([w7, w8])
        # not prune
        if i < normal_train:
            for Xtrain_t, ytrain_t in train_data:

                Xtrain_t = tf.reshape(Xtrain_t, shape=[-1, input_width, input_height, 1])
                grads = get_grad( Xtrain_t, ytrain_t)
                #total_loss += current_loss.numpy()
                apply_grad(opt,grads)

        if True:
            print("\n--------- epoch: ", i, " --------------")
            print_zero_stat(LENET5)
            print("total loss in this epoch: ", total_loss)
            ypred = tf.nn.softmax(LENET5(tf.constant(xtest)))
            ypred = tf.argmax(ypred, axis=1)
            tmp = tf.cast(tf.equal(ypred, ytest), tf.float32)
            acc = tf.reduce_mean(tmp).numpy()
            print("accuracy: ", acc)
            accuracy_values.append(acc)
            print("-------------------------------------")

            if i == normal_train - 1:
                root = tf.train.Checkpoint(
                    optimizer=opt, model=LENET5, optimizer_step=tf.compat.v1.train.get_or_create_global_step()
                )
                root.save("checkpoint-lenet5-before-pruning/ckpt")

            if i == total_train_epoch - 1:
                root = tf.train.Checkpoint(
                    optimizer=opt, model=LENET5, optimizer_step=tf.compat.v1.train.get_or_create_global_step()
                )
                root.save("checkpoint-lenet5-after-pruning/ckpt")

    return LENET5, accuracy_values


""" Quantization with kmeans """


def quantize(bits=5, cdfs=None, mode="linear"):

    """Performs the quantization and weight sharing with kmeans.

    Parameters
    ----------
    cdfs : tuple
        The cumulative distribution of weights for each layer of the neural network.
        It is required only for density mode
    bits : integer
        The maximum number of bits to use for storing the centroids : with n bits I can store 2^n different centroids
    mode : string
        The technique to initialize the centers for k-means

    How it works
    -------
    It assumes there is a checkpoint with the pruned model in the folder : /checkpoint-lenet300-after-pruning
        : previously train_with_pruning() needs to be called

    It loads the pruned model, get the original weights, get the quantized weights calling utility.get_quantized_weight(),
        and set the new weights.
    Finally, it stores in the folder /checkpoint-lenet300-after-quantization the quantized model

    Returns
    -------
    void

    Raises
    ------
    Exception('mode not found') if the mode passed is not in  {linear,density,forge,kmeans++}

    """

    # get the dataset
    [(xtrain, ytrain), (xval, yval), (xtest, ytest)] = utility.read_mnist(
        mnist_folder, flatten=True, num_train=-1
    )
    xtest = tf.reshape(xtest, shape=[-1, input_width, input_height, 1])
    ytest = tf.argmax(ytest, axis=1)

    # get the pruned NN
    net = LeNet5()
    opt = tf.keras.optimizers.Adam(0.001)
    root = tf.train.Checkpoint(
        optimizer=opt, model=net, optimizer_step=tf.compat.v1.train.get_or_create_global_step()
    )
    root.restore(tf.train.latest_checkpoint("checkpoint-lenet5-after-pruning"))

    ypred = tf.nn.softmax(net(tf.constant(xtest)))
    ypred = tf.argmax(ypred, axis=1)

    tmp = tf.cast(tf.equal(ypred, ytest), tf.float32)
    print("\naccuracy before quantization: ", tf.reduce_mean(tmp).numpy())

    # get the weights
    w1 = LENET5.conv1.get_weights()[0]
    w2 = LENET5.conv1.get_weights()[1]

    w3 = LENET5.conv2.get_weights()[0]
    w4 = LENET5.conv2.get_weights()[1]

    w5 = LENET5.dense.get_weights()[0]
    w6 = LENET5.dense.get_weights()[1]

    w7 = LENET5.logits.get_weights()[0]
    w8 = LENET5.logits.get_weights()[1]

    # get the quantized weights
    new_weight1, _ = utility.get_quantized_weight(w1, bits=bits, mode=mode, cdfs=(cdfs[0], cdfs[1]))
    new_weight2, _ = utility.get_quantized_weight(w2, bits=bits, mode=mode, cdfs=(cdfs[2], cdfs[3]))
    new_weight3, _ = utility.get_quantized_weight(w3, bits=bits, mode=mode, cdfs=(cdfs[4], cdfs[5]))
    new_weight4, _ = utility.get_quantized_weight(w4, bits=bits, mode=mode, cdfs=(cdfs[6], cdfs[7]))
    new_weight5, _ = utility.get_quantized_weight(w5, bits=bits, mode=mode, cdfs=(cdfs[8], cdfs[9]))
    new_weight6, _ = utility.get_quantized_weight(
        w6, bits=bits, mode=mode, cdfs=(cdfs[10], cdfs[11])
    )
    new_weight7, _ = utility.get_quantized_weight(
        w7, bits=bits, mode=mode, cdfs=(cdfs[12], cdfs[13])
    )
    new_weight8, _ = utility.get_quantized_weight(
        w8, bits=bits, mode=mode, cdfs=(cdfs[14], cdfs[15])
    )

    # set the new weights
    LENET5.conv1.set_weights([new_weight1, new_weight2])
    LENET5.conv2.set_weights([new_weight3, new_weight4])
    LENET5.dense.set_weights([new_weight5, new_weight6])
    LENET5.logits.set_weights([new_weight7, new_weight8])

    # print stats

    ypred = tf.nn.softmax(net(tf.constant(xtest)))
    ypred = tf.argmax(ypred, axis=1)

    # ytest = tf.argmax(ytest, axis=1)
    tmp = tf.cast(tf.equal(ypred, ytest), tf.float32)
    print("\naccuracy after quantization: ", tf.reduce_mean(tmp).numpy())

    # get the weights
    w1 = LENET5.conv1.get_weights()[0]
    w2 = LENET5.conv1.get_weights()[1]

    w3 = LENET5.conv2.get_weights()[0]
    w4 = LENET5.conv2.get_weights()[1]

    w5 = LENET5.dense.get_weights()[0]
    w6 = LENET5.dense.get_weights()[1]

    w7 = LENET5.logits.get_weights()[0]
    w8 = LENET5.logits.get_weights()[1]

    print("\nunique values :\n")
    print(len(np.unique(w1)))
    print(len(np.unique(w2)))
    print(len(np.unique(w3)))
    print(len(np.unique(w4)))
    print(len(np.unique(w5)))
    print(len(np.unique(w6)))
    print(len(np.unique(w7)))
    print(len(np.unique(w8)))

    root = tf.train.Checkpoint(
        optimizer=opt, model=net, optimizer_step=tf.compat.v1.train.get_or_create_global_step()
    )
    root.save("checkpoint-lenet5-after-quantization/ckpt")


def print_info():

    """Compute the cumulative weight distribution of the layers of the neural network
        and stores the images of the weight distribution in the folder /lenet300_report


    How it works
    -------
    1) load the model stored in /checkpoint-lenet300-before-pruning and call utility.show_weight_distribution()
        to store the plots of the weight of the nerual network before pruning
    2) load the model after pruning store in /checkpoint-lenet300-after-pruning and call utility.show_weight_distribution()
        to store the plots of the weight of the neural network after pruning and to obtain the cdf of each layer


    Returns
    -------
    tuple:
        it is a tuple where for each layer there is the x-values,and y-values for that layer

    """

    # before pruning

    [(xtrain, ytrain), (xval, yval), (xtest, ytest)] = utility.read_mnist(
        mnist_folder, flatten=True, num_train=-1
    )
    xtest = tf.reshape(xtest, shape=[-1, input_width, input_height, 1])
    ytest = tf.argmax(ytest, axis=1)

    # get the  NN before prune
    net = LeNet5()
    opt = tf.keras.optimizers.Adam(0.001)
    root = tf.train.Checkpoint(
        optimizer=opt, model=net, optimizer_step=tf.compat.v1.train.get_or_create_global_step()
    )
    root.restore(tf.train.latest_checkpoint("checkpoint-lenet5-before-pruning"))

    # DUMMY VARIABLE INITIALIZATION : OTHERWISE THE WEIGHTS ARE NOT RESTORED
    ypred = tf.nn.softmax(net(tf.constant(xtest)))
    ypred = tf.argmax(ypred, axis=1)

    tmp = tf.cast(tf.equal(ypred, ytest), tf.float32)
    # print('\nbefore pruning accuracy: ', tf.reduce_mean(tmp).numpy())

    w1 = LENET5.conv1.get_weights()[0]
    w2 = LENET5.conv1.get_weights()[1]
    w3 = LENET5.conv2.get_weights()[0]
    w4 = LENET5.conv2.get_weights()[1]
    w5 = LENET5.dense.get_weights()[0]
    w6 = LENET5.dense.get_weights()[1]
    w7 = LENET5.logits.get_weights()[0]
    w8 = LENET5.logits.get_weights()[1]

    utility.show_weight_distribution(w1, "lenet5_report/conv1-lenete5-weights.png")
    utility.show_weight_distribution(w2, "lenet5_report/conv1-lenete5-bias.png")
    utility.show_weight_distribution(w3, "lenet5_report/conv2-lenete5-weights.png")
    utility.show_weight_distribution(w4, "lenet5_report/conv2-lenete5-bias.png")
    utility.show_weight_distribution(w5, "lenet5_report/dense-lenete5-weights.png")
    utility.show_weight_distribution(w6, "lenet5_report/dense-lenete5-bias.png")
    utility.show_weight_distribution(w7, "lenet5_report/logits-lenete5-weights.png")
    utility.show_weight_distribution(w8, "lenet5_report/logits-lenete5-bias.png")

    # after pruning

    # get the  NN after prune
    net = LeNet5()
    opt = tf.keras.optimizers.Adam(0.001)
    root = tf.train.Checkpoint(
        optimizer=opt, model=net, optimizer_step=tf.compat.v1.train.get_or_create_global_step()
    )
    root.restore(tf.train.latest_checkpoint("checkpoint-lenet5-after-pruning"))

    # DUMMY VARIABLE INITIALIZATION : OTHERWISE THE WEIGHTS ARE NOT RESTORED
    ypred = tf.nn.softmax(net(tf.constant(xtest)))
    ypred = tf.argmax(ypred, axis=1)

    tmp = tf.cast(tf.equal(ypred, ytest), tf.float32)

    # print('\naccuracy after pruning: ', tf.reduce_mean(tmp).numpy())
    # print('\n zero stat:')
    # print_zero_stat(net)

    w1 = LENET5.conv1.get_weights()[0]
    w1 = w1.flatten()
    (idx,) = np.nonzero(w1 == 0)
    w1 = np.delete(w1, idx, axis=0)

    w2 = LENET5.conv1.get_weights()[1]
    w2 = w2.flatten()
    (idx,) = np.nonzero(w2 == 0)
    w2 = np.delete(w2, idx, axis=0)

    w3 = LENET5.conv2.get_weights()[0]
    w3 = w3.flatten()
    (idx,) = np.nonzero(w3 == 0)
    w3 = np.delete(w3, idx, axis=0)

    w4 = LENET5.conv2.get_weights()[1]
    w4 = w4.flatten()
    (idx,) = np.nonzero(w4 == 0)
    w4 = np.delete(w4, idx, axis=0)

    w5 = LENET5.dense.get_weights()[0]
    w5 = w5.flatten()
    (idx,) = np.nonzero(w5 == 0)
    w5 = np.delete(w5, idx, axis=0)

    w6 = LENET5.dense.get_weights()[1]
    w6 = w6.flatten()
    (idx,) = np.nonzero(w6 == 0)
    w6 = np.delete(w6, idx, axis=0)

    w7 = LENET5.logits.get_weights()[0]
    w7 = w7.flatten()
    (idx,) = np.nonzero(w7 == 0)
    w7 = np.delete(w7, idx, axis=0)

    w8 = LENET5.logits.get_weights()[1]
    w8 = w8.flatten()
    (idx,) = np.nonzero(w8 == 0)
    w8 = np.delete(w8, idx, axis=0)

    w1, w1_cdf = utility.show_weight_distribution(
        w1, "lenet5_report/conv1-lenete5-weights-after_pruning.png", plot_std=False, plot_cdf=True
    )
    w2, w2_cdf = utility.show_weight_distribution(
        w2, "lenet5_report/conv1-lenete5-bisd-after_pruning.png", plot_std=False, plot_cdf=True
    )
    w3, w3_cdf = utility.show_weight_distribution(
        w3, "lenet5_report/conv2-lenete5-weights-after_pruning.png", plot_std=False, plot_cdf=True
    )
    w4, w4_cdf = utility.show_weight_distribution(
        w4, "lenet5_report/conv2-lenete5-bias-after_pruning.png", plot_std=False, plot_cdf=True
    )
    w5, w5_cdf = utility.show_weight_distribution(
        w5, "lenet5_report/dense-lenete5-weights-after_pruning.png", plot_std=False, plot_cdf=True
    )
    w6, w6_cdf = utility.show_weight_distribution(
        w6, "lenet5_report/dense-lenete5-bias-after_pruning.png", plot_std=False, plot_cdf=True
    )
    w7, w7_cdf = utility.show_weight_distribution(
        w7, "lenet5_report/logits-lenete5-weights-after_pruning.png", plot_std=False, plot_cdf=True
    )
    w8, w8_cdf = utility.show_weight_distribution(
        w8, "lenet5_report/logits-lenete5-bias-after_pruning.png", plot_std=False, plot_cdf=True
    )

    return (
        w1,
        w1_cdf,
        w2,
        w2_cdf,
        w3,
        w3_cdf,
        w4,
        w4_cdf,
        w5,
        w5_cdf,
        w6,
        w6_cdf,
        w7,
        w7_cdf,
        w8,
        w8_cdf,
    )


std_smooth = True
thresholds = [1, 0.1, 1, 0.1, 0.5, 0, 0, 0]

normal_train = 1
prune_train = 2
semi_prune_train = 0

model, acc_nomal_train = train_with_pruning(
    thresholds, std_smooth, normal_train, prune_train, semi_prune_train
)

cdfs = print_info()
mode = "density"
bits = 5
quantize(cdfs=cdfs, mode=mode, bits=bits)
