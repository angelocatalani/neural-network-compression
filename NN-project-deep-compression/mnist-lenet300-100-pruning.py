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
    return l


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
    step = 0

    utility.download_mnist(mnist_folder)

    [(xtrain, ytrain), (xval, yval), (xtest, ytest)] = utility.read_mnist(mnist_folder, flatten=True, num_train=-1)

    net = LeNet300()

    loss_grad = tfe.implicit_value_and_gradients(loss)
    opt = tf.train.AdamOptimizer(0.001)

    root = tf.train.Checkpoint(optimizer=opt,
                               model=net,
                               optimizer_step=tf.train.get_or_create_global_step())
    root.restore(tf.train.latest_checkpoint('checkpoint-lenet300'))

    train_data = tf.data.Dataset.from_tensor_slices((xtrain, ytrain)).shuffle(1000).batch(32, drop_remainder=True)

    for i in range(55):

        # semi prune : the previous zeros are left but no zeros will be added
        total_loss = 0
        if (i >= 50):
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
        if (i >= 10 and i < 50):
            for Xtrain_t, ytrain_t in tfe.Iterator(train_data):
                # 1) get weight
                w1 = net.layer1.get_weights()[0]
                w2 = net.layer1.get_weights()[1]
                w3 = net.layer2.get_weights()[0]
                w4 = net.layer2.get_weights()[1]
                w5 = net.out.get_weights()[0]
                w6 = net.out.get_weights()[1]

                # 2) side effect on the weights under threshold values and get pruned indexes
                zero_idx1 = utility.prune_weigth(w1, std_smooth=False)
                zero_idx2 = utility.prune_weigth(w2, std_smooth=False)
                zero_idx3 = utility.prune_weigth(w3, std_smooth=False)
                zero_idx4 = utility.prune_weigth(w4, std_smooth=False)
                zero_idx5 = utility.prune_weigth(w5, std_smooth=False)
                zero_idx6 = utility.prune_weigth(w6, std_smooth=False)

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
        if (i < 10):
            for Xtrain_t, ytrain_t in tfe.Iterator(train_data):
                current_loss, grads = loss_grad(net, Xtrain_t, ytrain_t)
                total_loss += current_loss.numpy()
                opt.apply_gradients(grads)

        if (i % 2 == 0 or i >= 8 or True):
            print('\n--------- epoch: ', i, ' --------------')
            print_zero_stat(net)
            # ypred = net(tf.constant(Xtest))
            # ypred = tf.argmax(ypred, axis=1)
            # tmp = tf.cast(tf.equal(ypred, tf.constant(ytest.astype(np.int64))), tf.float32)
            # print('accuracy: ',tf.reduce_mean(tmp).numpy())
            print('total loss in this epoch: ', total_loss)
            # print_zero_stat(net)
            print('-------------------------------------')

            # root = tf.train.Checkpoint(optimizer=opt,
            #                           model=net,
            #                           optimizer_step=tf.train.get_or_create_global_step())
            # root.save('checkpoint-lenet300/ckpt')
            # root.restore(tf.train.latest_checkpoint('checkpoint-lenet300'))

            # step += 1

        # saver = tfe.Saver(net.variables)
        # saver.save('checkpoints/checkpoint.ckpt', global_step=step)
        # checkpoint_path = tf.train.latest_checkpoint('checkpoints')
        # saver.restore(checkpoint_path)
        # step+=1

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
    net = LeNet300()
    opt = tf.train.AdamOptimizer(0.001)
    root = tf.train.Checkpoint(optimizer=opt,
                               model=net,
                               optimizer_step=tf.train.get_or_create_global_step())
    root.restore(tf.train.latest_checkpoint('checkpoint-lenet300'))

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
    new_weight1,_ = utility.get_quantized_weight(w1, bits=4)
    new_weight2, _ = utility.get_quantized_weight(w2, bits=4)
    new_weight3, _ = utility.get_quantized_weight(w3, bits=4)
    new_weight4, _ = utility.get_quantized_weight(w4, bits=4)
    new_weight5, _ = utility.get_quantized_weight(w5, bits=4)
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
    print_zero_stat(net)



quantize()

# net=train_with_pruning()
