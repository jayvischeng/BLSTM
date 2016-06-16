import numpy as np
from numpy import *

from sklearn.metrics import roc_auc_score

def trapezoid_area(x1, x2, y1, y2):
    delta = abs(x2 - x1)
    return delta * 0.5 * (y1 + y2)

def get_auc(arr_score, arr_label, pos_label):
    score_label_list = []
    for index in xrange(len(arr_score)):
        score_label_list.append((float(arr_score[index]), int(arr_label[index])))
    score_label_list_sorted = sorted(score_label_list, key = lambda line:line[0], reverse = True)

    fp, tp = 0, 0
    lastfp, lasttp = 0, 0
    A = 0
    lastscore = None

    for score_label in score_label_list_sorted:
        score, label = score_label[:2]
        if score != lastscore:
            A += trapezoid_area(fp, lastfp, tp, lasttp)
            lastscore = score
            lastfp, lasttp = fp, tp
        if label == pos_label:
            tp += 1
        else:
            fp += 1

    A += trapezoid_area(fp, lastfp, tp, lasttp)
    A /= (fp * tp)
    return A
y_true1 = np.array([-1, -1, -1,-1, 1,  1, 1, 1, 1, 1])
y_scores1 = np.array([-1, -1, -1, 1, 1, -1, 1 , -1, 1, -1])
auc1 = roc_auc_score(y_true1, y_scores1)
#auc2 = get_auc(y_scores,y_true,0)
print(auc1)
y_true2 = np.array([1, 1, 1, 1, -1,  -1, -1, -1, -1, -1])
y_scores2 = np.array([1, 1, 1, -1, -1, 1, -1 , 1, -1, 1])
auc2 = roc_auc_score(y_true2, y_scores2)
print(auc2)