import category_encoders as ce
import numpy as np
import pandas as pd

class TargetEncoderMultiClass():
    def __init__(self, num_classes):
        self.ohe_encoder = ce.OneHotEncoder()
        self.te_encoders = [ce.TargetEncoder()]*(num_classes-1)
        self.fitted = False


    def fit(self, Z, y):
        y_onehot = self.ohe_encoder.fit_transform(y.astype(object))
        self.class_names = y_onehot.columns  # names of onehot encoded columns

        for num, class_ in enumerate(self.class_names[1:]):
            self.te_encoders[num].fit(Z.astype(object), y_onehot[class_])
        self.fitted = True

    def fit_transform(self, Z, y):
        y_onehot = self.ohe_encoder.fit_transform(y.astype(object))
        self.class_names = y_onehot.columns  # names of onehot encoded columns

        Z_te = pd.DataFrame(index=y.index)
        for num, class_ in enumerate(self.class_names[1:]):
            self.te_encoders[num].fit(Z.astype(object), y_onehot[class_])
            Z_te_c = self.te_encoders[num].transform(Z.astype(object), y_onehot[class_])
            Z_te_c.columns = [str(x) + '_' + str(class_) for x in Z_te_c.columns]
            Z_te = pd.concat([Z_te, Z_te_c], axis=1)

        self.fitted = True

        return Z_te

    def transform(self, Z, y):
        assert self.fitted == True, "Encoder not fitted!"
        y_onehot = self.ohe_encoder.transform(y.astype(object))
        self.class_names = y_onehot.columns  # names of onehot encoded columns

        Z_te = pd.DataFrame(index=y.index)
        for num, class_ in enumerate(self.class_names[1:]):
            Z_te_c = self.te_encoders[num].transform(Z.astype(object), y_onehot[class_])
            Z_te_c.columns = [str(x) + '_' + str(class_) for x in Z_te_c.columns]
            Z_te = pd.concat([Z_te, Z_te_c], axis=1)

        return Z_te

