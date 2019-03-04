import tensorflow.contrib.eager as tfe
import tensorflow as tf
import numpy as np
import utility


tf.enable_eager_execution()
mnist_folder = 'data/mnist'
layer1_size = 300
layer2_size = 100
input_size = 28 * 28
output_size = 10  # 1 hot encoder


normal_train=10
prune_train=10
semi_prune_train=1
total_train_epoch = normal_train+prune_train+semi_prune_train

''' Model '''


class LeNet300(tf.keras.Model):

    def __init__(self):
        super(LeNet300, self).__init__()
        self.layer1 = tf.layers.Dense(layer1_size, activation=tf.nn.relu)
        self.layer2 = tf.layers.Dense(layer2_size, activation=tf.nn.relu)
        self.out = tf.layers.Dense(output_size)

    def call(self, x, **kwargs):
        return self.out(self.layer2(self.layer1(x)))


# Define the loss function
def loss(net, x, y):
    ypred = net(x)
    l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=ypred))
    # Loss function with L2 Regularization with beta=0.01
    beta=0.01
    w1=net.layer1.get_weights()[0]
    w2 = net.layer2.get_weights()[0]
    w3 = net.out.get_weights()[0]
    regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)+ tf.nn.l2_loss(w3)
    loss = tf.reduce_mean(l + beta * regularizers)
    return loss


def print_zero_stat(net):
    [l1, l1_b] = net.layer1.get_weights()
    [l2, l2_b] = net.layer2.get_weights()
    [out, out_b] = net.out.get_weights()
    print('Layer1 zeros:', np.count_nonzero(l1 == 0), ' on ', l1.shape[0] * l1.shape[1])
    print('Layer1 bias zeros :', np.count_nonzero(l1_b == 0), ' on ', l1_b.shape[0])
    print('Layer2 zeros :', np.count_nonzero(l2 == 0), ' on ', l2.shape[0] * l2.shape[1])
    print('Layer2 bias zeros :', np.count_nonzero(l2_b == 0), ' on ', l2.shape[0])
    print('Out zeros :', np.count_nonzero(out == 0), ' on ', out.shape[0] * out.shape[1])
    print('Out bias zeros :', np.count_nonzero(out_b == 0), ' on ', out_b.shape[0])


'''Train + Pruning'''


def train_with_pruning():

    utility.download_mnist(mnist_folder)

    [(xtrain, ytrain), (xval, yval), (xtest, ytest)] = utility.read_mnist(mnist_folder, flatten=True, num_train=-1)
    ytest = tf.argmax(ytest, axis=1)


    net = LeNet300()

    loss_grad = tfe.implicit_value_and_gradients(loss)
    opt = tf.train.AdamOptimizer(0.001)

    '''
    root = tf.train.Checkpoint(optimizer=opt,
                               model=net,
                               optimizer_step=tf.train.get_or_create_global_step())
    root.restore(tf.train.latest_checkpoint('checkpoint-lenet300'))
    '''

    train_data = tf.data.Dataset.from_tensor_slices((xtrain, ytrain)).shuffle(1000).batch(32, drop_remainder=True)

    for i in range(total_train_epoch):

        # semi prune : the previous zeros are left but no zeros will be added
        total_loss = 0
        if (i >=normal_train+prune_train):
            for Xtrain_t, ytrain_t in tfe.Iterator(train_data):
                current_loss, grads = loss_grad(net, Xtrain_t, ytrain_t)
                total_loss += current_loss.numpy()
                opt.apply_gradients(grads)
                w1 = net.layer1.get_weights()[0]
                w2 = net.layer1.get_weights()[1]
                w3 = net.layer2.get_weights()[0]
                w4 = net.layer2.get_weights()[1]
                w5 = net.out.get_weights()[0]
                w6 = net.out.get_weights()[1]

                '''
                zero_idx1 = w1 == 0
                zero_idx2 = w2 == 0
                zero_idx3 = w3 == 0
                zero_idx4 = w4 == 0
                zero_idx5 = w5 == 0
                zero_idx6 = w6 == 0
                '''

                w1[zero_idx1] = 0
                w2[zero_idx2] = 0
                w3[zero_idx3] = 0
                w4[zero_idx4] = 0
                w5[zero_idx5] = 0
                w6[zero_idx6] = 0
                net.layer1.set_weights([w1, w2])
                net.layer2.set_weights([w3, w4])
                net.out.set_weights([w5, w6])

        # prune
        if (i >= normal_train and i < normal_train+prune_train):
            for Xtrain_t, ytrain_t in tfe.Iterator(train_data):
                # 1) get weight
                w1 = net.layer1.get_weights()[0]
                w2 = net.layer1.get_weights()[1]
                w3 = net.layer2.get_weights()[0]
                w4 = net.layer2.get_weights()[1]
                w5 = net.out.get_weights()[0]
                w6 = net.out.get_weights()[1]

                # 2) side effect on the weights under threshold values and get pruned indexes
                zero_idx1 = utility.prune_weigth(w1, threshold=1)
                zero_idx2 = utility.prune_weigth(w2, threshold=0.1)
                zero_idx3 = utility.prune_weigth(w3, threshold=1)
                zero_idx4 = utility.prune_weigth(w4, threshold=0.1)
                zero_idx5 = utility.prune_weigth(w5, threshold=0.5)
                zero_idx6 = utility.prune_weigth(w6, threshold=0)

                # 3) set the neural network pruned weights
                net.layer1.set_weights([w1, w2])
                net.layer2.set_weights([w3, w4])
                net.out.set_weights([w5, w6])

                # 4) compute and apply the gradient optimization on the pruned neural network
                current_loss, grads = loss_grad(net, Xtrain_t, ytrain_t)
                total_loss += current_loss.numpy()

                opt.apply_gradients(grads)

                # 5) get the updated weights and zeroed the previously pruned weights
                w1 = net.layer1.get_weights()[0]
                w2 = net.layer1.get_weights()[1]
                w3 = net.layer2.get_weights()[0]
                w4 = net.layer2.get_weights()[1]
                w5 = net.out.get_weights()[0]
                w6 = net.out.get_weights()[1]

                w1[zero_idx1] = 0
                w2[zero_idx2] = 0
                w3[zero_idx3] = 0
                w4[zero_idx4] = 0
                w5[zero_idx5] = 0
                w6[zero_idx6] = 0
                net.layer1.set_weights([w1, w2])
                net.layer2.set_weights([w3, w4])
                net.out.set_weights([w5, w6])
        # not prune
        if (i < normal_train):
            for Xtrain_t, ytrain_t in tfe.Iterator(train_data):
                current_loss, grads = loss_grad(net, Xtrain_t, ytrain_t)
                total_loss += current_loss.numpy()
                opt.apply_gradients(grads)

        if (True):
            print('\n--------- epoch: ', i, ' --------------')
            print_zero_stat(net)
            print('total loss in this epoch: ', total_loss)

            ypred = tf.nn.softmax(net(tf.constant(xtest)))
            ypred = tf.argmax(ypred, axis=1)


            tmp = tf.cast(tf.equal(ypred, ytest), tf.float32)
            print('\naccuracy: ', tf.reduce_mean(tmp).numpy())

            print('-------------------------------------')

            if (i == normal_train-1):
                root = tf.train.Checkpoint(optimizer=opt,
                                           model=net,
                                           optimizer_step=tf.train.get_or_create_global_step())
                root.save('checkpoint-lenet300-before-pruning/ckpt')
                root.restore(tf.train.latest_checkpoint('checkpoint-lenet300-before-pruning'))

            if (i == total_train_epoch-1):
                root = tf.train.Checkpoint(optimizer=opt,
                                           model=net,
                                           optimizer_step=tf.train.get_or_create_global_step())
                root.save('checkpoint-lenet300-after-pruning/ckpt')




    return net


''' Quantization with kmeans '''


# assume there is a checkpoint with the trained and pruned model : before  train_with_pruning has been called
def quantize(cdfs=None,bits = 5,mode='linear'):

    # get the dataset
    [(xtrain, ytrain), (xval, yval), (xtest, ytest)] = utility.read_mnist(mnist_folder, flatten=True, num_train=-1)

    # get the pruned NN
    net = LeNet300()
    opt = tf.train.AdamOptimizer(0.001)
    root = tf.train.Checkpoint(optimizer=opt,
                               model=net,
                               optimizer_step=tf.train.get_or_create_global_step())
    root.restore(tf.train.latest_checkpoint('checkpoint-lenet300-after-pruning'))

    ypred = tf.nn.softmax(net(tf.constant(xtest)))
    ypred = tf.argmax(ypred, axis=1)

    ytest = tf.argmax(ytest, axis=1)
    tmp = tf.cast(tf.equal(ypred, ytest), tf.float32)
    print('before quantization \naccuracy: ', tf.reduce_mean(tmp).numpy())
    print_zero_stat(net)

    # get the weights
    w1 = net.layer1.get_weights()[0]
    w2 = net.layer1.get_weights()[1]

    w3 = net.layer2.get_weights()[0]
    w4 = net.layer2.get_weights()[1]

    w5 = net.out.get_weights()[0]
    w6 = net.out.get_weights()[1]

    # get the quantized weights
    new_weight1,_ = utility.get_quantized_weight(w1, bits=bits,mode=mode,cdfs=(cdfs[0],cdfs[1]))
    new_weight2, _ = utility.get_quantized_weight(w2, bits=bits,mode=mode,cdfs=(cdfs[2],cdfs[3]))
    new_weight3, _ = utility.get_quantized_weight(w3, bits=bits,mode=mode,cdfs=(cdfs[4],cdfs[5]))
    new_weight4, _ = utility.get_quantized_weight(w4, bits=bits,mode=mode,cdfs=(cdfs[6],cdfs[7]))
    new_weight5, _ = utility.get_quantized_weight(w5, bits=bits,mode=mode,cdfs=(cdfs[8],cdfs[9]))
    #new_weight6, _ = utility.get_quantized_weight(w6, bits=4) -> not enough elements

    # set the new weights
    net.layer1.set_weights([new_weight1, new_weight2])
    net.layer2.set_weights([new_weight3, new_weight4])
    net.out.set_weights([new_weight5, w6])

    # print stats
    ypred = tf.nn.softmax(net(tf.constant(xtest)))
    ypred = tf.argmax(ypred, axis=1)
    tmp = tf.cast(tf.equal(ypred, ytest), tf.float32)
    print('after quantization \naccuracy: ', tf.reduce_mean(tmp).numpy())

    # get the weights
    w1 = net.layer1.get_weights()[0]
    w2 = net.layer1.get_weights()[1]

    w3 = net.layer2.get_weights()[0]
    w4 = net.layer2.get_weights()[1]

    w5 = net.out.get_weights()[0]
    w6 = net.out.get_weights()[1]

    print('unique values :\n')
    print(len(np.unique(w1)))
    print(len(np.unique(w2)))
    print(len(np.unique(w3)))
    print(len(np.unique(w4)))
    print(len(np.unique(w5)))
    print(len(np.unique(w6)))


    root = tf.train.Checkpoint(optimizer=opt,
                               model=net,
                               optimizer_step=tf.train.get_or_create_global_step())
    root.save('checkpoint-lenet300-after-quantization/ckpt')

def print_info():


    # before pruning

    [(xtrain, ytrain), (xval, yval), (xtest, ytest)] = utility.read_mnist(mnist_folder, flatten=True, num_train=-1)
    ytest = tf.argmax(ytest, axis=1)


    # get the  NN before prune
    net = LeNet300()
    opt = tf.train.AdamOptimizer(0.001)
    root = tf.train.Checkpoint(optimizer=opt,
                               model=net,
                               optimizer_step=tf.train.get_or_create_global_step())
    root.restore(tf.train.latest_checkpoint('checkpoint-lenet300-before-pruning'))

    #DUMMY VARIABLE INITIALIZATION : OTHERWISE THE WEIGHTS ARE NOT RESTORED
    ypred = tf.nn.softmax(net(tf.constant(xtest)))
    ypred = tf.argmax(ypred, axis=1)


    tmp = tf.cast(tf.equal(ypred, ytest), tf.float32)
    print('\naccuracy: ', tf.reduce_mean(tmp).numpy())



    w1 = net.layer1.get_weights()[0]
    w2 = net.layer1.get_weights()[1]
    w3 = net.layer2.get_weights()[0]
    w4 = net.layer2.get_weights()[1]
    w5 = net.out.get_weights()[0]
    w6 = net.out.get_weights()[1]

    utility.show_weight_distribution(w1,'lenet300_report/layer1-lenete300-weights.png')
    utility.show_weight_distribution(w2, 'lenet300_report/layer1-lenete300-bias.png')
    utility.show_weight_distribution(w3, 'lenet300_report/layer2-lenete300-weights.png')
    utility.show_weight_distribution(w4, 'lenet300_report/layer2-lenete300-bias.png')
    utility.show_weight_distribution(w5, 'lenet300_report/out-lenete300-weights.png')
    utility.show_weight_distribution(w6, 'lenet300_report/out-lenete300-bias.png')

    # after pruning

    # get the  NN before prune
    net = LeNet300()
    opt = tf.train.AdamOptimizer(0.001)
    root = tf.train.Checkpoint(optimizer=opt,
                               model=net,
                               optimizer_step=tf.train.get_or_create_global_step())
    root.restore(tf.train.latest_checkpoint('checkpoint-lenet300-after-pruning'))

    #DUMMY VARIABLE INITIALIZATION : OTHERWISE THE WEIGHTS ARE NOT RESTORED
    ypred = tf.nn.softmax(net(tf.constant(xtest)))
    ypred = tf.argmax(ypred, axis=1)

    tmp = tf.cast(tf.equal(ypred, ytest), tf.float32)

    print('\naccuracy after pruning: ', tf.reduce_mean(tmp).numpy())
    print_zero_stat(net)



    w1 = net.layer1.get_weights()[0]
    w1=w1.flatten()
    idx,=np.nonzero(w1==0)
    w1=np.delete(w1,idx,axis=0)




    w2 = net.layer1.get_weights()[1]
    w2=w2.flatten()
    idx,=np.nonzero(w2==0)
    w2=np.delete(w2,idx,axis=0)


    w3 = net.layer2.get_weights()[0]
    w3=w3.flatten()
    idx,=np.nonzero(w3==0)
    w3=np.delete(w3,idx,axis=0)


    w4 = net.layer2.get_weights()[1]
    w4=w4.flatten()
    idx,=np.nonzero(w4==0)
    w4=np.delete(w4,idx,axis=0)


    w5 = net.out.get_weights()[0]
    w5=w5.flatten()
    idx,=np.nonzero(w5==0)
    w5=np.delete(w5,idx,axis=0)


    w6 = net.out.get_weights()[1]
    w6=w6.flatten()
    idx,=np.nonzero(w6==0)
    w6=np.delete(w6,idx,axis=0)

    w1,w1_cdf=utility.show_weight_distribution(w1,'lenet300_report/layer1-lenete300-weights-after_pruning.png',plot_std=False,plot_cdf=True)
    w2, w2_cdf =utility.show_weight_distribution(w2, 'lenet300_report/layer1-lenete300-bias-after_pruning.png',plot_std=False,plot_cdf=True)
    w3, w3_cdf =utility.show_weight_distribution(w3, 'lenet300_report/layer2-lenete300-weights-after_pruning.png',plot_std=False,plot_cdf=True)
    w4, w4_cdf =utility.show_weight_distribution(w4, 'lenet300_report/layer2-lenete300-bias-after_pruning.png',plot_std=False,plot_cdf=True)
    w5, w5_cdf =utility.show_weight_distribution(w5, 'lenet300_report/out-lenete300-weights-after_pruning.png',plot_std=False,plot_cdf=True)
    w6, w6_cdf =utility.show_weight_distribution(w6, 'lenet300_report/out-lenete300-bias-after_pruning.png',plot_std=False,plot_cdf=True)

    return (w1,w1_cdf,w2,w2_cdf,w3,w3_cdf,w4,w4_cdf,w5,w5_cdf,w6,w6_cdf)


#train_with_pruning()
cdfs=print_info()

quantize(cdfs=cdfs,mode='linear')


