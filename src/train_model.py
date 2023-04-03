import matplotlib.pyplot as plt

from settings import (
    BATCH_SIZE,
    EPOCHS
)

def _plot_history(history):
    """
    Plots accuracy/loss for as a function of epochs
    """

    fig, axs = plt.subplots(2)

    # accuracy plot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy")

    # error plot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error")

    plt.show()

def train_model(model, x_train, x_validation, x_test, y_train, y_validation, y_test, model_name=None, plot_history=False):
    """
    Train model with given data splits
    """

    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=BATCH_SIZE, epochs=EPOCHS)

    if plot_history:
        _plot_history(history)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc}, test loss: {test_loss}")

    if model_name is not None:
        model.save(f"models\\{model_name}")

    return model

