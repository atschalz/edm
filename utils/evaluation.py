import numpy as np

from sklearn.metrics import accuracy_score as acc
from tensorflow.keras.metrics import CategoricalAccuracy as cat_acc
from tensorflow_addons.metrics import F1Score
from sklearn.metrics import roc_auc_score as auroc
from sklearn.metrics import f1_score as f1
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

def get_metrics(y_true,y_pred,target):
    if target == "binary":
        # Inputs: True=vector with binary labels, Pred=vector with class 1 probabilitues
        res = {
            "Accuracy": acc(y_true, np.round(y_pred)),
            "AUROC": auroc(y_true, y_pred),
            "F1": f1(y_true, np.round(y_pred))
        }
    elif target == "continuous":
        # Inputs: True=numerical vector, Pred=numerical vector
        res = {
           "MSE": mse(y_true, y_pred),
           "R2": r2(y_true, y_pred)
              }

    elif target == "categorical":
        # Inputs: True=N*C OHE matrix, Pred=N*C Softmax matrix
        res = {
            "Accuracy": cat_acc()(y_true, y_pred).numpy(),
            "F1": F1Score(num_classes=y_true.shape[1], average="macro")(y_true, y_pred).numpy(),
            "AUROC": auroc(y_true, y_pred, multi_class="ovo", average="macro")
                       }
    return res





