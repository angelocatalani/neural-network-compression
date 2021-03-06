from neural_network_compression import utility
from neural_network_compression.le_net_300_100_trainer import LeNet300100Trainer
import tensorflow as tf
import matplotlib.pyplot as plt
from neural_network_compression.trainer import LeNetDataset

mnist_folder = "data/mnist"

utility.download_mnist(mnist_folder)

[(x_train, y_train), (x_validation, y_validation), (x_test, y_test)] = utility.read_mnist(
    mnist_folder, flatten=True, num_train=-1
)
y_test = tf.argmax(y_test, axis=1)

TRAIN_EPOCHS = 200
PRUNE_TRAIN_EPOCHS = 300
SEMI_PRUNE_TRAIN_EPOCHS = 1

if __name__ == "__main__":
    train_dataset = LeNetDataset(x_train, y_train)
    test_dataset = LeNetDataset(x_test, y_test)
    trainer = LeNet300100Trainer()

    train_accuracies = trainer.train(
        train_dataset=train_dataset, test_dataset=test_dataset, epochs=TRAIN_EPOCHS
    )
    pruned_train_accuracies = trainer.pruned_train(
        train_dataset=train_dataset, test_dataset=test_dataset, epochs=SEMI_PRUNE_TRAIN_EPOCHS
    )
    semi_pruned_train_accuracies = trainer.semi_pruned_train(
        train_dataset=train_dataset, test_dataset=test_dataset, epochs=PRUNE_TRAIN_EPOCHS
    )

    fig, ax = plt.subplots()
    plt.legend(["Epochs"])
    ax.plot(range(TRAIN_EPOCHS), train_accuracies, color="r", label="train")
    ax.plot(
        range(TRAIN_EPOCHS, TRAIN_EPOCHS + SEMI_PRUNE_TRAIN_EPOCHS),
        pruned_train_accuracies,
        color="g",
        label="pruned train",
    )
    ax.plot(
        range(
            TRAIN_EPOCHS + SEMI_PRUNE_TRAIN_EPOCHS,
            TRAIN_EPOCHS + SEMI_PRUNE_TRAIN_EPOCHS + PRUNE_TRAIN_EPOCHS,
        ),
        semi_pruned_train_accuracies,
        color="b",
        label="semi pruned train",
    )
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()
