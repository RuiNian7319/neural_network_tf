"""
Evaluation Module.  Used to add upper and lower limits to the neural network regression for Valve position vs.
Pressure Ratio

By: Rui Nian

Last Edit:  January 3rd, 2018

Methods:
    limit_adder: Plots the valve position vs. pressure ratio, and then plots the most probable outcome using
                 the neural network.  Then, the upper and lower limits will be plotted.
     live_plots: Plots the pressure ratio as a function of time on the top plot and the valve position with the top and
                 bottom limits on the bottom plot.  Any points outside the band is considered anomalous
"""

import numpy as np
import matplotlib.pyplot as plt

import gc


def limit_adder(x, y, predictions, axis='pressure'):
    """
    Comments:  The valve_pos outlier labelling is not tuned properly.


    Inputs
    ----
              x: Feature vectors
              y: Labels, y
    predictions: Predicted labels, y_hat
           axis: To generate limits based on pressure or valve position

    Variables
    ----
          upper: Upper bound
          lower: Lower bound
    """
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

    else:
        raise ValueError("Incorrect axis.  Choices are valve_pos and pressure.  You inputted {}.".format(axis))

    plt.scatter(x, y, color='grey')
    plt.scatter(x, upper, s=6, color='red')
    plt.scatter(x, lower, s=6, color='red')
    plt.scatter(x, predictions, color='green')
    plt.xlim([0.2, 1])
    plt.ylim([0, 70])
    plt.show()


def live_plots(pres_ratio, valve_pos, predictions, time_start=0, time_end=100, axis='pressure'):
    """
    Comments:  The valve_pos outlier labelling is not tuned properly.


    Inputs
    ----
     pres_ratio: Feature vectors
      valve_pos: Labels, y
    predictions: Predicted labels, y_hat
     time_start: Start time of plot
       time_end: End time of plot
           axis: To generate limits based on pressure or valve position

    Variables
    ----
          upper: Upper bound
          lower: Lower bound
    """
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
        for i, value in enumerate(pres_ratio):
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

    else:
        raise ValueError("Incorrect axis.  Choices are valve_pos and pressure.  You inputted {}.".format(axis))

    time_axis = np.linspace(time_start, time_end, time_end - time_start)

    plt.subplot(2, 1, 1)
    plt.title('Pressure Ratio')
    plt.xlabel('Time')
    plt.ylabel('Pressure Ratio')
    plt.scatter(time_axis, pres_ratio[time_start:time_end], color='blue')

    plt.subplot(2, 1, 2)
    plt.title('Valve Position')
    plt.xlabel('Time')
    plt.ylabel('Pressure Ratio')
    plt.scatter(time_axis, valve_pos[time_start:time_end], color='green')
    plt.scatter(time_axis, upper[time_start:time_end], color='red')
    plt.scatter(time_axis, lower[time_start:time_end], color='red')

    plt.show()
