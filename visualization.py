# File: visualization.py
# File Created: Saturday, 5th January 2019 12:41:42 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import matplotlib.pyplot as plt

import models


def plot_loss(train_loss, validation_loss=None, fname=None):
    """
    Log-log plot of the loss versus iterations.
    """
    plt.figure()
    plt.loglog(train_loss[0], train_loss[1])
    legend = ["Training"]
    if validation_loss is not None:
        plt.loglog(validation_loss[0], validation_loss[1])
        legend.append("Validation")
    plt.xlabel("Minibatch")
    plt.ylabel("RMSE")
    plt.legend(legend)
    if fname is not None:
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()


def plot_prediction(model, data, y=None, title=None, fname=None, window=None):
    """
    Use the model to predict the output, then plot the input, prediction, and
    target all together.
    """
    x, t = data.x, data.y
    if window is not None:
        x, t = x[window[0]: window[1]], t[window[0]: window[1]]
    if y is None:
        y = model.predict(x)
    plt.figure(figsize=(12, 4))
    plt.plot(x)
    plt.plot(t)
    plt.plot(y)
    legend = ['Input', 'Target', 'Prediction']
    if isinstance(model, models.Autoregressive):
        # Plot a vertical line at the first data point where a full, non-padded
        # input is available.  Before this point, the model doesn't have all of 
        # the input that it relies on to predict the output, so we shouldn't
        # expect a perfect match.
        ylim = plt.ylim()
        plt.plot((model.input_length - 1,) * 2, ylim, "r--")
        legend.append("Full input")
    plt.legend(legend)
    if title is not None:
        plt.title(title)
    if fname is not None:
        print("Saving to {}...".format(fname))
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()
