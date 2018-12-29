import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def suncor_early_pred(predictions, labels, early_window, num_of_events, threshold=0.5):

    """     DOES NOT WORK IF 1ST EXAMPLE IS AN EVENT !!
            predictions:  Predictions made by logistic regression
            actual_data:  Actual Suncor pipeline data with labels on first column
           early_window:  How big the window is for early detection
          num_of_events:  Total events in the data set
              threshold:  Threshold for rounding a number up

        To use: recall_overall, precision, not_detected, misfired, detection_list = \
                suncor_early_pred(predictions, train_y, 100, 47, 0.5)
    """
    # Convert to boolean
    predictions = np.round(predictions + 0.5 - threshold)

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

    error = 0
    misfired = []

    for i, event in enumerate(predictions):
        # If the prediction is positive
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
