import numpy as np
from sklearn.utils.multiclass import type_of_target

def train_nb(X,y):
    unique_labels, label_counts = np.unique(y, return_counts=True)
    label_p_dict = {}
    for label in unique_labels:
        sample_idx = [i for i in range(len(y)) if y[i] == label]
        prior = (sample_idx / len(y)).astype(float)
        Xc = X[sample_idx]
        dict_list = []
        for attr_idx in range(Xc.shape[1]):
            p_dict = {}
            # if continuous, store 𝜇 and 𝜎
            if type_of_target(Xc[:, attr_idx]) == 'unknown':
                p_dict['mu'] = np.mean(Xc[:,attr_idx])
                p_dict['sigma'] = np.var(Xc[:,attr_idx])
            # if multi-class, store a attr-prob dictionary
            else:
                unique_attrs = np.unique(X[:, attr_idx])
                for attr in unique_attrs:
                    # p_dict[attr] = ((Xc[:,attr_idx]==attr).sum())/(Xc.shape[0])
                    # smoothing
                    p_dict[attr] = ((Xc[:, attr_idx] == attr).sum() + 1) / (Xc.shape[0] + len(unique_attrs))
            dict_list.append((attr_idx, p_dict))
        label_p_dict[label] = (prior, dict_list)

def predict_nb(label_p_dict, X):
    preds = []
    attr_continuous = [
        True if type_of_target(X[:,i])=='unknown' else False
        for i in range(X.shape[1])
    ]
    for i in range(len(X)):
        pred_dict = {}
        sample = X[i]
        for label, (prior, label_dict) in label_p_dict.items():
            p = prior
            for attr_idx, p_dict in label_dict:
                if attr_continuous[attr_idx]:
                    mu = p_dict['mu']
                    sigma = p_dict['sigma']
                    p *= 1.0/(np.sqrt(2*np.pi)*sigma) * np.exp(-((sample[attr_idx]-mu)**2)/(2*(sigma**2)))
                else:
                    p *= p_dict[sample[attr_idx]]
            pred_dict[label] = p
        preds.append(pred_dict)




