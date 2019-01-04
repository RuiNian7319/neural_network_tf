"""
Evaluation code for Neural Network Classification.

By: Rui Nian

The following definitions are in the script:
    Suncor_early_pred: Evaluates the precision and recall of the ML model on all Suncor data
           live_plots: Plots 2 figures.  Top is prediction in plant, bottom is live plant data
               scaler: Scales predictions by the scaling factor
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import gc


def suncor_early_pred(predictions, labels, early_window, num_of_events, threshold=0.5):

    """     DOES NOT WORK IF 1ST EXAMPLE IS AN EVENT !!
            predictions:  Predictions made by logistic regression
            actual_data:  Actual Suncor pipeline data with labels on first column
           early_window:  How big the window is for early detection
          num_of_events:  Total events in the data set
              threshold:  Threshold for rounding a number up

        To use: recall_overall, precision, not_detected, misfired, detection_list = \
                suncor_early_pred(predictions, train_y, 70, 47, 0.5)
    """
    # Convert to boolean
    predictions = np.round(predictions + 0.5 - threshold)

    """
    Recall calculations
    """

    detected = 0
    did_detect = []
    not_detected = []

    for i, event in enumerate(labels):
        # If an event occurs
        if event == 1 and (labels[i - 1] != 1 or i == 0):
            # If predictions detected the event up to event, the event is considered as "detected"
            if 1 in predictions[i - early_window:i + 1]:
                detected += 1
                did_detect.append(i)
            else:
                not_detected.append(i)

    recall_overall = detected / num_of_events

    """
    Precision calculations
    """

    error = 0
    misfired = []

    for i, event in enumerate(predictions):
        # If the prediction is positive and was not positive in the previous 20 steps
        if event == 1 and 1 not in predictions[i - 20:i]:
            # If there is an actual event, nothing happens
            if 1 in labels[i:i + early_window]:
                pass
            # If there is no event, add to error
            else:
                error += 1
                misfired.append(i)

    precision = detected / (error + detected)

    return recall_overall, precision, not_detected, misfired, did_detect


def live_plots(prediction, real_value, start, end, y=0.1):

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


def scaler(data, scale=26):
    """
    Inputs
    ----
                data: Data to be scaled
               scale: Amount to multiply all predictions by

    Returns
    ----
                data: Scaled predictions: min(data * scale, 1)
    """

    for i, value in enumerate(data):
        if value >= 0.02:
            data[i] = min(data[i] * scale, 1)

    return data
