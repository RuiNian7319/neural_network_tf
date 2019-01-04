import numpy as np
import matplotlib.pyplot as plt


def plots(prediction, real_value, start, end, y=0.1):

    """
    Plots the real plant trajectory vs the predicted

    Inputs
         -----
         perdiction:  Prediction from machine learning model
         real_value:  Real value from the data set
              start:  Start index of the plot
                end:  End index of the plot
                  y:  Cut off threshold to round up
    """
    plt.subplot(2, 1, 1)
    plt.xlabel("Time")
    plt.ylabel("Percent below Threshold, %")
    plt.step(np.linspace(0, end - start, end - start), prediction[start:end])

    plt.axhline(y=y, c='r', linestyle='--')

    plt.subplot(2, 1, 2)
    plt.xlabel("Time")
    plt.ylabel("Plant Data")
    plt.step(np.linspace(0, end - start, end - start), real_value[start:end], c='r')

    plt.show()
