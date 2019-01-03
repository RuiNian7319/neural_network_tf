"""
Evaluation Module.  Used to add upper and lower limits to the neural network regression for Valve position vs.
Pressure Ratio

By: Rui Nian

Last Edit:  January 3rd, 2018
"""

import numpy as np
import matplotlib.pyplot as plt

import gc


def limit_adder(x, y, predictions, axis='pressure'):
    upper = np.zeros(predictions.shape)
    lower = np.zeros(predictions.shape)

    if axis == "valve_pos":
        for i, pred in enumerate(predictions):
            if pred >= 40:
                upper[i] = pred + 3
                lower[i] = pred - 3
            elif 30 <= pred < 40:
                upper[i] = pred + 4.5
                lower[i] = pred - 4.5
            elif 20 <= pred < 30:
                upper[i] = pred + 6
                lower[i] = pred - 6
            elif 6 <= pred < 20:
                upper[i] = pred + 9
                lower[i] = pred - 9
            else:
                print("Value outside range")

    elif axis == "pressure":
        for i, value in enumerate(x):
            if value >= 0.88:
                upper[i] = predictions[i] + 5
                lower[i] = predictions[i] - 5
            elif 0.68 <= value < 0.88:
                upper[i] = predictions[i] + 3
                lower[i] = predictions[i] - 4.5
            elif 0.56 <= value < 0.68:
                upper[i] = predictions[i] + 3.5
                lower[i] = predictions[i] - 3.5
            elif 0.43 <= value < 0.56:
                upper[i] = predictions[i] + 3
                lower[i] = predictions[i] - 9
            elif 0.32 <= value < 0.43:
                upper[i] = predictions[i] + 6
                lower[i] = predictions[i] - 9
            elif 0.2 <= value < 0.32:
                upper[i] = predictions[i] + 5
                lower[i] = predictions[i] - 5
            else:
                print("Value outside range")

    plt.scatter(x, y, color='grey')
    plt.scatter(x, upper, s=6, color='red')
    plt.scatter(x, lower, s=6, color='red')
    plt.scatter(x, predictions, color='green')
    plt.xlim([0.2, 1])
    plt.ylim([0, 70])
    plt.show()
