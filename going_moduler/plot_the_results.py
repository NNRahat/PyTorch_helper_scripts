import matplotlib.pyplot as plt
from typing import Tuple,List,Dict
def plot_curves(model_0: Dict[str, list[float]],
                model_1:Dict[str, list[float]] = None):
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]

    train_acc = results["train_acc"]
    test_acc = results["test_acc"]


    epochs = range(len(model_1["train_loss"]))

    plt.figure(figsize = (15,7))

    if not model_2:
        plt.subplot(1,2,1)
        plt.plot(epochs, model_0["train_loss"], label = "train loss")
        plt.plot(epochs, model_0["test_loss"], label = "test_loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(epochs, model_1["train_acc"] , label="train accuracy")
        plt.plot(epochs, model_1["test_acc"] , label= "test accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()

    if model_2:
        plt.subplot(2,2,1)
        plt.plot(epochs, model_0["train_loss"], label="train_loss")
        plt.plot(epochs, model_1["test_loss"], label="test_loss")
        plt.title("Train_loss")
        plt.legend()
        plt.subplot(2,2,2)
        plt.plot(epochs, model_0["train_acc"], label="train_acc")
        plt.plot(epochs, model_1["test_acc"], label="test_acc")
        plt.title("Train_loss")
        plt.legend()
        plt.subplot(2,2,3)
        plt.plot(epochs, model_0["train_loss"], label="train_loss")
        plt.plot(epochs, model_1["test_loss"], label="test_loss")
        plt.title("Train_loss")
        plt.legend()
        plt.subplot(2,2,4)
        plt.plot(epochs, model_0["train_acc"], label="train_acc")
        plt.plot(epochs, model_1["test_acc"], label="test_acc")
        plt.title("Train_loss")
        plt.legend()


