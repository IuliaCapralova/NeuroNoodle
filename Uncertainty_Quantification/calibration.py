import numpy as np
import matplotlib.pyplot as plt


def calibration(y_true, y_pred, y_unc, num_bins=10, conf_type="maxprob"):
    '''
    Divide predictions by confidence values into bins, get accuracy for
    each confidence level and plot a reliability plot.
    Includes ECE (expected calibration error).

    Args:
    y_true = n one-hot encoded target vectors
    y_pred = n probability vectors
    y_unc = n entropy measures
    num_bins = number of confidence bins to split the range of confidences into
    conf_type = measurement of confidence (maximum probability or entropy)
    '''

    if conf_type == "maxprob":
        conf = np.max(y_pred, axis=1)
    if conf_type == "entropy":
        conf = [1 - ent for ent in y_unc]

    pred_onehot = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    predictions = []
    for i in range(len(y_true)):
        predictions.append([conf[i], pred_onehot[i], y_true[i]])

    # create a set of confidence ranges for bins
    conf_ranges = np.linspace(0.0, 1.0, num_bins+1)

    # create bins for each confidence level, each bin will include
    # all the predictions with the confidence level of the bin
    bins = [[] for _ in range(num_bins)]
    # pred is [conf, pred_onehot, y_true]
    for pred in predictions:
        bin_idx = int(pred[0]*num_bins)
        if bin_idx == num_bins:
            bin_idx = num_bins-1
        bins[bin_idx].append(pred)

    # get average accuracy for each bin
    bin_accs = []
    ece = 0
    for i,bin in enumerate(bins):
        if len(bin) == 0:
            correct_pred_freq = 0
        else:
            correct_pred_freq = sum(pred[1]==pred[2] for pred in bin)/len(bin)
        bin_accs.append(correct_pred_freq)
        # ece is weighted by the proportion of samples in this bin with
        # respect to the total number of samples
        cur_conf = i * 1/num_bins + 1/(2*num_bins)
        ece += (len(bin)/len(y_pred)) * abs(correct_pred_freq - cur_conf)

    reliability_plot(conf_ranges, bin_accs, ece)


def reliability_plot(conf_ranges, bin_accs, ece):
    '''
    Creates a reliability plot (confidence vs accuracy) with ECE.
    '''
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.hist(conf_ranges[:-1], bins=conf_ranges, weights=bin_accs, edgecolor='black')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Reliability Plot (ECE={ece})')
    plt.legend()
    plt.show()