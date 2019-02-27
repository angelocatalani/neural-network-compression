import tensorflow.contrib.eager as tfe
import tensorflow as tf
import numpy as np
from sklearn import datasets, model_selection
from utility import prune_weigth,get_iris_dataset
tf.enable_eager_execution()




# hidden layer is 4x300
# hidden layer bias is a list of 300 elements
# out_layer is 300x100
# out_layer_bias is a list of 100 elements



hidden_layer_bias_size  = 10
out_layer_bias_size = 4

hidden_layer_size = 4*hidden_layer_bias_size
out_layer_size = hidden_layer_bias_size*out_layer_bias_size


class NN(tf.keras.Model):

    def __init__(self):
        super(NN, self).__init__()
        self.hidden = tf.layers.Dense(hidden_layer_bias_size , activation=tf.nn.tanh)
        self.out = tf.layers.Dense(out_layer_bias_size)

    def call(self, x, **kwargs):
        return self.out(self.hidden(x))


# Define the loss function
def loss(net,x, y):
    ypred = net(x)
    l = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=ypred)
    return l

def train_no_pruning():
    Xtrain, Xtest, ytrain, ytest = get_iris_dataset()

    net = NN()
    loss_grad = tfe.implicit_gradients(loss)
    opt = tf.train.AdamOptimizer(100)

    # all data in a single step
    Xtrain_t = tf.constant(Xtrain)
    ytrain_t = tf.constant(ytrain)

    for i in range(5):
        grads = loss_grad(net,Xtrain_t, ytrain_t)
        opt.apply_gradients(grads)
        print('epoch: ',i)

    ypred = net(tf.constant(Xtest))
    ypred = tf.argmax(ypred, axis=1)
    tmp = tf.cast(tf.equal(ypred, tf.constant(ytest.astype(np.int64))), tf.float32)
    print('accuracy: ',tf.reduce_mean(tmp).numpy())

    [hidden_layer,hidden_layer_bias] = net.hidden.get_weights()
    [out_layer , out_layer_bias]= net.out.get_weights()
    # hidden layer is 4x15
    # hidden layer bias is a list of 15 elements
    # out_layer is 15x3
    # out_layer_bias is a list of 3 elements

    print( np.count_nonzero(hidden_layer==0) )
    print(np.count_nonzero(hidden_layer_bias == 0))
    print(np.count_nonzero(out_layer == 0))
    print(np.count_nonzero(out_layer_bias == 0))




def print_zero_stat(net):
    [hidden_layer,hidden_layer_bias] = net.hidden.get_weights()

    [out_layer , out_layer_bias]= net.out.get_weights()
    print('hidden layer zeros :',np.count_nonzero(hidden_layer==0),' on ',hidden_layer_size)
    print('bias hidden layer zeros :',np.count_nonzero(hidden_layer_bias == 0),' on ',hidden_layer_bias_size)
    print('output  layer zeros :',np.count_nonzero(out_layer == 0),' on ',out_layer_size)
    print('bias output  layer zeros :',np.count_nonzero(out_layer_bias == 0),' on ',out_layer_bias_size)



def train_with_pruning():


    Xtrain, Xtest, ytrain, ytest = get_iris_dataset()

    net = NN()
    loss_grad = tfe.implicit_value_and_gradients(loss)
    opt = tf.train.AdamOptimizer(0.001)

    # all data in a single step
    #Xtrain_t = tf.constant(Xtrain)
    #ytrain_t = tf.constant(ytrain)
    train_data = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain)).batch(32,drop_remainder=True)

    for i in range(10000):
        # prune
        if (i>300  ):
           for Xtrain_t, ytrain_t in tfe.Iterator(train_data):

                # 1) get weight
                w1 = net.hidden.get_weights()[0]
                w2 = net.hidden.get_weights()[1]
                w3 = net.out.get_weights()[0]
                w4 = net.out.get_weights()[1]

                # 2) side effect on the weights under threshold values and get pruned indexes
                zero_idx1 =prune_weigth(w1)
                zero_idx2 =prune_weigth(w2)
                zero_idx3 = prune_weigth(w3)
                zero_idx4 = prune_weigth(w4)

                # 3) set the neural network pruned weights
                net.hidden.set_weights([w1, w2])
                net.out.set_weights([w3, w4])

                # 4) compute and apply the gradient optimization on the pruned neural network
                current_loss,grads = loss_grad(net, Xtrain_t, ytrain_t)

                opt.apply_gradients(grads)

                # 5) get the updated weights and zeroed the previously pruned weights
                w1 = net.hidden.get_weights()[0]
                w2 = net.hidden.get_weights()[1]
                w3 = net.out.get_weights()[0]
                w4 = net.out.get_weights()[1]

                w1[zero_idx1] = 0
                w2[zero_idx2] = 0
                w3[zero_idx3] = 0
                w4[zero_idx4] = 0
                net.hidden.set_weights([w1, w2])
                net.out.set_weights([w3, w4])
        # not prune
        if (i<=300):
            for Xtrain_t, ytrain_t in tfe.Iterator(train_data):

                current_loss,grads = (current_loss,[(grad_hidden, weight_hidden), (grad_bias_hidden, bias_hidden),
                         (grad_output, weight_out), (grad_bias_output, bias_output)] )= loss_grad(net, Xtrain_t, ytrain_t)
                opt.apply_gradients(grads)
        if (  i%100 == 0 or i==10):
            print('\n--------- epoch: ',i,' --------------')
            #ypred = net(tf.constant(Xtest))
            #ypred = tf.argmax(ypred, axis=1)
            #tmp = tf.cast(tf.equal(ypred, tf.constant(ytest.astype(np.int64))), tf.float32)
            #print('accuracy: ',tf.reduce_mean(tmp).numpy())
            print('current loss: ',current_loss)
            print_zero_stat(net)
            print('-------------------------------------')
    return net



net = train_with_pruning()
Xtrain, Xtest, ytrain, ytest = get_iris_dataset()
print_zero_stat(net)
ypred = net(tf.constant(Xtest))
ypred = tf.argmax(ypred, axis=1)
tmp = tf.cast(tf.equal(ypred, tf.constant(ytest.astype(np.int64))), tf.float32)
print('accuracy: ', tf.reduce_mean(tmp).numpy())
print_zero_stat(net)