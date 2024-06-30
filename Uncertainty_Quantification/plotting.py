import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

def confmatrix(y_true, y_pred, normalized=True):        
    cm = confusion_matrix(y_true, y_pred)

    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmtt = '.2%'
    else:
        fmtt = 'd'

    annot = np.empty_like(cm).astype(str)
    n_classes = cm.shape[0]
    for i in range(n_classes):
        for j in range(n_classes):
            if i == 0 and j == 0:
                annot[i, j] = f'TN\n{cm[i, j]:{fmtt}}'
            elif i == 0 and j == 1:
                annot[i, j] = f'FP\n{cm[i, j]:{fmtt}}'
            elif i == 1 and j == 0:
                annot[i, j] = f'FN\n{cm[i, j]:{fmtt}}'
            elif i == 1 and j == 1:
                annot[i, j] = f'TP\n{cm[i, j]:{fmtt}}'

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=['B', 'M'], yticklabels=['B', 'M'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_entropies(id_epi, ood_epi, labels=['ID', 'OOD']):
    bins = np.arange(0, 1.1, 0.1)

    plt.hist(id_epi, bins=bins, color='red', alpha=0.5, label=labels[0])
    plt.hist(ood_epi, bins=bins, color='blue', alpha=0.5, label=labels[1])

    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.title('Entropy in ID vs OOD')
    plt.legend()
    plt.show()