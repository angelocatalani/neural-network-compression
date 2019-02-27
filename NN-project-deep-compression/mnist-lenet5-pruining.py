import tensorflow.contrib.eager as tfe
import tensorflow as tf
import numpy as np
import utility
tf.enable_eager_execution()
mnist_folder = 'data/mnist'

input_width=28
input_height=28
input_size=input_height*input_width
filter1_size=20
filter2_size=50
dense_size = 256
output_size=10


normal_train=2
prune_train=0
semi_prune_train=0
total_train_epoch = normal_train+prune_train+semi_prune_train


''' Model '''
class LeNet5(tf.keras.Model):

    def __init__(self):
        super(LeNet5, self).__init__()

        # weight: (5,5,1,20)=500 + (20,)
        self.conv1 = tf.layers.Conv2D(filters=filter1_size, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        self.pool1 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)

        # weight: (5,5,20,50)=25000 + (50,)
        self.conv2 = tf.layers.Conv2D(filters=filter2_size, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        self.pool2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)

        #  weight: (2450, 256)= 627200 + (256,)
        self.dense = tf.layers.Dense(dense_size, activation=tf.nn.relu)
        self.dropout = tf.layers.Dropout(0.2)

        # weight: (256, 10) = 2560 + (10,)
        self.logits = tf.layers.Dense(units=output_size)

    def call(self, x,training=True):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        feature_dim = x.shape[1]*x.shape[2]*x.shape[3]
        x = tf.reshape(x, (-1,feature_dim))
        x = self.dense(x)
        x = self.dropout(x, training=training)
        return self.logits(x)

# Define the loss function
def loss(net, x, y):
    ypred = net(x)
    l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=ypred))
    return l

def print_zero_stat(net):
    [l1,l1_b] = net.conv1.get_weights()
    [l2 , l2_b]= net.conv2.get_weights()
    [d, d_b] = net.dense.get_weights()
    [l, l_b] = net.logits.get_weights()

    print('Conv1 zeros:',np.count_nonzero(l1==0),' on ',l1.shape[0]*l1.shape[1]*l1.shape[2]*l1.shape[3])
    print('Conv1 bias zeros :',np.count_nonzero(l1_b == 0),' on ',l1_b.shape[0])
    print('Conv2 zeros :',np.count_nonzero(l2 == 0),' on ',l2.shape[0]*l2.shape[1]*l2.shape[2]*l2.shape[3])
    print('Conv2 bias zeros :',np.count_nonzero(l2_b == 0),' on ',l2_b.shape[0])
    print('Dense zeros :',np.count_nonzero(d == 0),' on ',d.shape[0]*d.shape[1])
    print('Dense bias zeros :',np.count_nonzero(d_b == 0),' on ',d_b.shape[0])
    print('Logits zeros :',np.count_nonzero(l == 0),' on ',l.shape[0]*l.shape[1])
    print('Logits bias zeros :',np.count_nonzero(l_b == 0),' on ',l_b.shape[0])


''' Train + Pruning'''
def train_with_pruning():
    step = 0
    utility.download_mnist(mnist_folder)
    [(xtrain, ytrain), (xval, yval), (xtest, ytest)] = utility.read_mnist(mnist_folder, flatten=True,num_train=-1)
    net = LeNet5()
    loss_grad = tfe.implicit_value_and_gradients(loss)
    opt = tf.train.AdamOptimizer(0.001)

    train_data = tf.data.Dataset.from_tensor_slices((xtrain, ytrain)).shuffle(1000).batch(32, drop_remainder=True)

    for i in range(total_train_epoch ):

        # semi prune : the previous zeros are left but no zeros will be added
        total_loss=0
        if (i>=normal_train+prune_train):

            for Xtrain_t, ytrain_t in tfe.Iterator(train_data):
                Xtrain_t = tf.reshape(Xtrain_t, shape=[-1, input_width, input_height, 1])
                current_loss, grads = loss_grad(net, Xtrain_t, ytrain_t)
                total_loss+=current_loss.numpy()
                opt.apply_gradients(grads)

                #w1 = net.conv1.get_weights()[0]
                #w2 = net.conv1.get_weights()[1]

                #w3 = net.conv2.get_weights()[0]
                #w4 = net.conv2.get_weights()[1]

                w5 = net.dense.get_weights()[0]
                w6 = net.dense.get_weights()[1]

                w7 = net.logits.get_weights()[0]
                w8 = net.logits.get_weights()[1]

                #w1[zero_idx1] = 0
                #w2[zero_idx2] = 0
                #w3[zero_idx3] = 0
                #w4[zero_idx4] = 0
                w5[zero_idx5] = 0
                w6[zero_idx6] = 0
                w7[zero_idx7] = 0
                w8[zero_idx8] = 0

                #net.conv1.set_weights([w1, w2])
                #net.conv2.set_weights([w3, w4])
                net.dense.set_weights([w5, w6])
                net.logits.set_weights([w7, w8])

        # prune
        if (i >= normal_train and i<normal_train+prune_train):
            for Xtrain_t, ytrain_t in tfe.Iterator(train_data):
                Xtrain_t = tf.reshape(Xtrain_t, shape=[-1, input_width, input_height, 1])

                # 1) get weight
                #w1 = net.conv1.get_weights()[0]
                #w2 = net.conv1.get_weights()[1]

                #w3 = net.conv2.get_weights()[0]
                #w4 = net.conv2.get_weights()[1]

                w5 = net.dense.get_weights()[0]
                w6 = net.dense.get_weights()[1]

                w7 = net.logits.get_weights()[0]
                w8 = net.logits.get_weights()[1]

                # 2) side effect on the weights under threshold values and get pruned indexes

                #zero_idx1 = utility.prune_weigth(w1)
                #zero_idx2 = utility.prune_weigth(w2)
                #zero_idx3 = utility.prune_weigth(w3)
                #zero_idx4 = utility.prune_weigth(w4)
                zero_idx5 = utility.prune_weigth(w5,std_smooth=True,threshold=1)
                zero_idx6 = utility.prune_weigth(w6,std_smooth=True,threshold=1)
                zero_idx7 = utility.prune_weigth(w7,std_smooth=True,threshold=1)
                zero_idx8 = utility.prune_weigth(w8,std_smooth=True,threshold=1)


                # 3) set the neural network pruned weights
                #net.conv1.set_weights([w1, w2])
                #net.conv2.set_weights([w3, w4])
                net.dense.set_weights([w5, w6])
                net.logits.set_weights([w7, w8])


                # 4) compute and apply the gradient optimization on the pruned neural network
                current_loss, grads = loss_grad(net, Xtrain_t, ytrain_t)
                total_loss += current_loss.numpy()

                opt.apply_gradients(grads)

                # 5) get the updated weights and zeroed the previously pruned weights
                #w1 = net.conv1.get_weights()[0]
                #w2 = net.conv1.get_weights()[1]

                #w3 = net.conv2.get_weights()[0]
                #w4 = net.conv2.get_weights()[1]

                w5 = net.dense.get_weights()[0]
                w6 = net.dense.get_weights()[1]

                w7 = net.logits.get_weights()[0]
                w8 = net.logits.get_weights()[1]

                #w1[zero_idx1] = 0
                #w2[zero_idx2] = 0
                #w3[zero_idx3] = 0
                #w4[zero_idx4] = 0
                w5[zero_idx5] = 0
                w6[zero_idx6] = 0
                w7[zero_idx7] = 0
                w8[zero_idx8] = 0

                #net.conv1.set_weights([w1, w2])
                #net.conv2.set_weights([w3, w4])
                net.dense.set_weights([w5, w6])
                net.logits.set_weights([w7, w8])
        # not prune
        if (i < normal_train):
            for Xtrain_t, ytrain_t in tfe.Iterator(train_data):

                Xtrain_t=tf.reshape(Xtrain_t, shape=[-1, input_width, input_height, 1])
                current_loss, grads = loss_grad(net, Xtrain_t,ytrain_t)
                total_loss += current_loss.numpy()
                opt.apply_gradients(grads)


        if (i % 2 == 0 or i >= 8 or i<=3 or True):
            print('\n--------- epoch: ', i, ' --------------')
            print_zero_stat(net)
            print('total loss in this epoch: ', total_loss)
            print('-------------------------------------')
            root = tf.train.Checkpoint(optimizer=opt,
                                       model=net,
                                       optimizer_step=tf.train.get_or_create_global_step())
            root.save('checkpoint-lenet5/ckpt')



    xtest= tf.reshape(xtest, shape=[-1, input_width, input_height, 1])
    ypred = tf.nn.softmax(net(tf.constant(xtest)))
    ypred = tf.argmax(ypred, axis=1)

    ytest = tf.argmax(ytest, axis=1)
    tmp = tf.cast(tf.equal(ypred, ytest), tf.float32)
    print('accuracy: ', tf.reduce_mean(tmp).numpy())
    print_zero_stat(net)

    return net



''' Quantization with kmeans '''


# assume there is a checkpoint with the trained and pruned model : before  train_with_pruning has been called
def quantize(bits = 5):

    # get the dataset
    [(xtrain, ytrain), (xval, yval), (xtest, ytest)] = utility.read_mnist(mnist_folder, flatten=True, num_train=-1)

    # get the pruned NN
    net = LeNet5()
    opt = tf.train.AdamOptimizer(0.001)
    root = tf.train.Checkpoint(optimizer=opt,
                               model=net,
                               optimizer_step=tf.train.get_or_create_global_step())
    root.restore(tf.train.latest_checkpoint('checkpoint-lenet5'))

    xtest= tf.reshape(xtest, shape=[-1, input_width, input_height, 1])
    ypred = tf.nn.softmax(net(tf.constant(xtest)))
    ypred = tf.argmax(ypred, axis=1)

    ytest = tf.argmax(ytest, axis=1)
    tmp = tf.cast(tf.equal(ypred, ytest), tf.float32)
    print('accuracy before quantization: ', tf.reduce_mean(tmp).numpy())
    print_zero_stat(net)


    # get the weights
    w1 = net.conv1.get_weights()[0]
    w2 = net.conv1.get_weights()[1]

    w3 = net.conv2.get_weights()[0]
    w4 = net.conv2.get_weights()[1]

    w5 = net.dense.get_weights()[0]
    w6 = net.dense.get_weights()[1]

    w7 = net.logits.get_weights()[0]
    w8 = net.logits.get_weights()[1]

    # get the quantized weights
    new_weight1,_ = utility.get_quantized_weight(w1, bits=4)
    #new_weight2, _ = utility.get_quantized_weight(w2, bits=4)
    new_weight3, _ = utility.get_quantized_weight(w3, bits=4)
    #new_weight4, _ = utility.get_quantized_weight(w4, bits=4)
    new_weight5, _ = utility.get_quantized_weight(w5, bits=4)
    #new_weight6, _ = utility.get_quantized_weight(w6, bits=4) -> not enough elements
    new_weight7, _ = utility.get_quantized_weight(w7, bits=4)
    #new_weight8, _ = utility.get_quantized_weight(w8, bits=4)

    # set the new weights
    net.conv1.set_weights([new_weight1, w2])
    net.conv2.set_weights([new_weight3, w4])
    net.dense.set_weights([new_weight5, w6])
    net.logits.set_weights([new_weight7, w8])

    # print stats

    ypred = tf.nn.softmax(net(tf.constant(xtest)))
    ypred = tf.argmax(ypred, axis=1)

    #ytest = tf.argmax(ytest, axis=1)
    tmp = tf.cast(tf.equal(ypred, ytest), tf.float32)
    print('accuracy after quantization: ', tf.reduce_mean(tmp).numpy())


#train_with_pruning()


quantize()





