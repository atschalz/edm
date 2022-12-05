import sys
import os
import pandas as pd
import numpy as np
import arff # make sure to pip install liac-arff
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from category_encoders.glmm import GLMMEncoder
from sklearn.model_selection import KFold

import pickle


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def process_dataset(dataset_name, target="", mode="train_val_test", RS=42, hct=10, test_ratio=0.2, val_ratio=0.1, folds=5):
    if not os.path.exists(f"../data/prepared/{dataset_name}"):
        os.mkdir(f"../data/prepared/{dataset_name}")

    new_path = f"{mode}_RS{RS}_hct{hct}"
    if mode == "cv":
        new_path += f"_{folds}folds"
    elif mode == "train_test":
        new_path += f"_split{1-test_ratio*100}-{test_ratio*100}"
    elif mode == "train_val_test":
        new_path += f"_split{round(100-(test_ratio+val_ratio)*100)}-{round(test_ratio*100)}-{round(val_ratio*100)}"

    if not os.path.exists(f"../data/prepared/{dataset_name}/"+new_path):
        os.mkdir(f"../data/prepared/{dataset_name}/"+new_path)

    return_dict = {}

    # Load datasets
    if dataset_name=="hussain":
        dataset = arff.load(open(f"../data/raw/{dataset_name}/Sapfile1.arff", 'rt'))
        df = pd.DataFrame(dataset['data'], columns=[i[0] for i in dataset["attributes"]])
    else:
        df = pd.read_excel(f"../data/raw/{dataset_name}/{dataset_name}.xlsx")


    # Drop columns with more than 5% missings (difference to Pargent et al.)
    df = df.drop(df.columns[df.isna().sum() / df.shape[0] > 0.95], axis=1)

    # Define dataset-specific column types
    if dataset_name=="academic_performance":
        df = df.drop(["COD_S11", "Cod_SPro"], axis=1)
        alternative_targets = ["CR_PRO", "QR_PRO", "CC_PRO", "WC_PRO", "FEP_PRO", "ENG_PRO", "QUARTILE", "PERCENTILE",
                               "2ND_DECILE", ]
        df = df.drop(alternative_targets, axis=1)

        z_cols = list(df.columns[list(np.logical_and(df.nunique() >= hct, df.dtypes == "object"))])
        bin_cols = list(df.columns[df.nunique() == 2])
        cat_cols = list(set(df.columns[df.dtypes == "object"]) - set(z_cols + bin_cols))
        numeric_cols = ["SEL", "SEL_IHE", "MAT_S11", "CR_S11", "CC_S11", "BIO_S11", "ENG_S11"]
        y_col = "G_SC"
    if dataset_name=="hussain":
        # Many ordinal features - treat as categorical for now, except of target, which is treated as numeric
        df = df.drop(["ms"],axis=1)
        y_col = "esp"
        df.esp[df.esp=="Pass"] = 0
        df.esp[df.esp=="Good"] = 1
        df.esp[df.esp=="Vg"] = 2
        df.esp[df.esp=="Best"] = 3
        df.esp = df.esp.astype(np.float32)
        z_cols = list(df.columns[list(np.logical_and(df.nunique() >= hct, df.dtypes == "object"))])
        bin_cols = list(df.columns[df.nunique() == 2])
        cat_cols = list(set(df.columns[df.dtypes == "object"]) - set(z_cols + bin_cols))
        numeric_cols = []


    else:
        # 1. Identify target
        y_col = "Class"
        # 2. Identify binary columns = zwei Ausprägungen
        bin_cols = list(set(df.nunique()[df.nunique() == 2].index) - set([y_col]))
        # 3. Identify high cardinality = dytpes==object & >hct Ausprägunge
        z_cols = list(df.nunique()[np.logical_and(df.nunique() >= hct, df.dtypes == "object")].index)
        # 4. Identify cat cols = Rest dytpes==object
        cat_cols = list(set(df.dtypes[df.dtypes == "object"].index) - set([y_col] + bin_cols + z_cols))
        # 5. Rest is numeric
        numeric_cols = list(set(df.columns[df.dtypes != "object"]) - set([y_col]) - set(bin_cols))

    # label encode categorical features
    for col in df.columns[df.dtypes == "object"]:
        le_ = LabelEncoder()
        df[col] = le_.fit_transform(df[col].astype(str))

    assert len(cat_cols+[y_col]+z_cols+bin_cols+numeric_cols)==df.shape[1], "Column type definitions imply different dimensionality than dataset"

    return_dict["y_col"] = y_col
    return_dict["cat_cols"] = cat_cols
    return_dict["bin_cols"] = bin_cols
    return_dict["z_cols"] = z_cols

    # Split data and target
    y = df[y_col]
    X = df.drop(y_col, axis=1)

    if mode=="cv":
        kf = KFold(n_splits=folds, shuffle=True, random_state=RS)
        split = kf.split(X, y)
    elif mode in ["train_test", "train_val_test"]:
        test_indices = X.sample(frac=test_ratio, random_state=RS).index
        split = [(np.array(list(set(X.index).difference(test_indices))), np.array(test_indices))]

    for num, (train_indices, test_indices) in enumerate(split):
        X_train = X.loc[train_indices]
        y_train = y.loc[train_indices]
        X_test = X.loc[test_indices]
        y_test = y.loc[test_indices]
        if mode in ["train_val_test", "cv"]:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=RS, shuffle=True)

        if mode in ["train_test", "train_val_test"]:
            str_num = ""
        elif mode == "cv":
            str_num = f"_{num}"

        # Pargent Pipeline
        ### Imputation 1
        # Recode categorical column missings as NA
        if len(cat_cols + z_cols) > 0:
            for col in cat_cols + z_cols:
                X_train[col].loc[X_train[col].isna()] = -1
                if mode in ["train_val_test", "cv"]:
                    X_val[col].loc[X_val[col].isna()] = -1
                X_test[col].loc[X_test[col].isna()] = -1

        # Impute binary columns with train mode
        bin_impute = {}
        if len(bin_cols) > 0:
            for col in bin_cols:
                u, c = np.unique(X_train[col][~X_train[col].isna()], return_counts=True)
                bin_impute[col] = u[np.argmax(c)]
                X_train[col].loc[X_train[col].isna()] = bin_impute[col]
                if mode in ["train_val_test", "cv"]:
                    X_val[col].loc[X_val[col].isna()] = bin_impute[col]
                X_test[col].loc[X_test[col].isna()] = bin_impute[col]

        # Impute continuous columns with train mean & standardize
        cont_impute = {}
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                cont_impute[col] = X_train[col][~X_train[col].isna()].mean()
                X_train[col].loc[X_train[col].isna()] = cont_impute[col]
                if mode in ["train_val_test", "cv"]:
                    X_val[col].loc[X_val[col].isna()] = cont_impute[col]
                X_test[col].loc[X_test[col].isna()] = cont_impute[col]

            # Standardize
            scaler = StandardScaler()
            # fit and transform scaler on X_train and X_test
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            if mode in ["train_val_test", "cv"]:
                X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

        # Standardize continuous targets
        if target =="continuous":
            target_scaler = StandardScaler()
            # fit and transform scaler on X_train and X_test
            y_train = pd.Series(target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel(),index=X_train.index)
            if mode in ["train_val_test", "cv"]:
                y_val = pd.Series(target_scaler.transform(y_val.values.reshape(-1, 1)).ravel(),index=X_val.index)
            y_test = pd.Series(target_scaler.transform(y_test.values.reshape(-1, 1)).ravel(),index=X_test.index)

            return_dict["target_scaler"] = target_scaler

        ### Encoding & Imputation 2
        # get encodings for high-card cat features
        z_ohe_encoded_train = pd.DataFrame(index=X_train.index)
        if mode in ["train_val_test", "cv"]:
            z_ohe_encoded_val = pd.DataFrame(index=X_val.index)
        z_ohe_encoded_test = pd.DataFrame(index=X_test.index)
        z_target_encoded_train = pd.DataFrame(index=X_train.index)
        if mode in ["train_val_test", "cv"]:
            z_target_encoded_val = pd.DataFrame(index=X_val.index)
        z_target_encoded_test = pd.DataFrame(index=X_test.index)
        if len(z_cols) > 0:
            for col in z_cols:
                # OHE
                u, c = np.unique(X_train[col], return_counts=True)
                collapse_ = u[c < hct]
                col_collapsed_train = X_train[col].apply(lambda x: -2 if x in collapse_ else x)
                if mode in ["train_val_test", "cv"]:
                    col_collapsed_val = X_val[col].apply(lambda x: -2 if x in collapse_ else x)
                col_collapsed_test = X_test[col].apply(lambda x: -2 if x in collapse_ else x)
                enc = OneHotEncoder(handle_unknown='ignore')
                ohe_encoded_train = pd.DataFrame(enc.fit_transform(pd.DataFrame(col_collapsed_train)).toarray(),
                                                 columns=enc.get_feature_names([col]), index=X_train.index)
                if mode in ["train_val_test", "cv"]:
                    ohe_encoded_val = pd.DataFrame(enc.transform(pd.DataFrame(col_collapsed_val)).toarray(),
                                               columns=enc.get_feature_names([col]), index=X_val.index)
                ohe_encoded_test = pd.DataFrame(enc.transform(pd.DataFrame(col_collapsed_test)).toarray(),
                                                columns=enc.get_feature_names([col]), index=X_test.index)
                z_ohe_encoded_train = z_ohe_encoded_train.join(ohe_encoded_train)
                if mode in ["train_val_test", "cv"]:
                    z_ohe_encoded_val = z_ohe_encoded_val.join(ohe_encoded_val)
                z_ohe_encoded_test = z_ohe_encoded_test.join(ohe_encoded_test)

                # Target encoding
                encoder = TargetEncoder()
                re_encoded_train = encoder.fit_transform(X_train[col].astype(object), y_train)
                if mode in ["train_val_test", "cv"]:
                    re_encoded_val = encoder.transform(X_val[col].astype(object), y_val)
                re_encoded_test = encoder.transform(X_test[col].astype(object), y_test)
                z_target_encoded_train = z_target_encoded_train.join(re_encoded_train)
                if mode in ["train_val_test", "cv"]:
                    z_target_encoded_val = z_target_encoded_val.join(re_encoded_val)
                z_target_encoded_test = z_target_encoded_test.join(re_encoded_test)

            # GLMM encoding (scales poorly by samples (no. of RE does not really matter): 721ms for 100, 9.04s for 500, 59.8s for 1000)
        #     encoder = GLMMEncoder()
        #     re_encoded_train = encoder.fit_transform(X_train[z_cols].astype(object), y_train)
        #     re_encoded_val = encoder.fit_transform(X_val[col].astype(object), y_val)
        #     re_encoded_test = encoder.fit_transform(X_test[col].astype(object), y_test)
        #     z_target_encoded_train = z_target_encoded_train.join(re_encoded_train)
        #     z_target_encoded_val = z_target_encoded_val.join(re_encoded_val)
        #     z_target_encoded_test = z_target_encoded_test.join(re_encoded_test)


        ### Drop constants (Drop features that are constant during training. As none of the original datasets includes constant columns, this step only removes constant features that are produced by the encoders or the CV splitting procedure)
        if any(X_train.nunique() == 1):
            drop_cols = X_train.columns[X_train.nunique() == 1]
            X_train = X_train.drop(drop_cols, axis=1)
            if mode in ["train_val_test", "cv"]:
                X_val = X_val.drop(drop_cols, axis=1)
            X_test = X_test.drop(drop_cols, axis=1)
        if any(z_ohe_encoded_train.nunique() == 1):
            drop_cols = z_ohe_encoded_train.columns[z_ohe_encoded_train.nunique() == 1]
            z_ohe_encoded_train = z_ohe_encoded_train.drop(drop_cols, axis=1)
            if mode in ["train_val_test", "cv"]:
                z_ohe_encoded_val = z_ohe_encoded_val.drop(drop_cols, axis=1)
            z_ohe_encoded_test = z_ohe_encoded_test.drop(drop_cols, axis=1)
        if any(z_target_encoded_train.nunique() == 1):
            drop_cols = z_target_encoded_train.columns[z_target_encoded_train.nunique() == 1]
            z_target_encoded_train = z_target_encoded_train.drop(drop_cols, axis=1)
            if mode in ["train_val_test", "cv"]:
                z_target_encoded_val = z_target_encoded_val.drop(drop_cols, axis=1)
            z_target_encoded_test = z_target_encoded_test.drop(z_target_encoded_test.columns[z_target_encoded_test.nunique() == 1], axis=1)

        return_dict["z_ohe_encoded_train"+str_num] = z_ohe_encoded_train
        if mode in ["train_val_test", "cv"]:
            return_dict["z_ohe_encoded_val"+str_num] = z_ohe_encoded_val
        return_dict["z_ohe_encoded_test"+str_num] = z_ohe_encoded_test
        return_dict["z_target_encoded_train"+str_num] = z_target_encoded_train
        if mode in ["train_val_test", "cv"]:
            return_dict["z_target_encoded_val"+str_num] = z_target_encoded_val
        return_dict["z_target_encoded_test"+str_num] = z_target_encoded_test


        ### Final one-hot-ecoding
        # Encode low-card cat features
        if len(cat_cols) > 0:
            enc = OneHotEncoder(handle_unknown='ignore')

            encoded_train = pd.DataFrame(enc.fit_transform(X_train[cat_cols]).toarray(),
                                         columns=enc.get_feature_names(cat_cols), index=X_train.index)
            X_train.drop(columns=cat_cols, inplace=True)
            X_train = X_train.join(encoded_train)

            if mode in ["train_val_test", "cv"]:
                encoded_val = pd.DataFrame(enc.transform(X_val[cat_cols]).toarray(),
                                           columns=enc.get_feature_names(cat_cols), index=X_val.index)
                X_val.drop(columns=cat_cols, inplace=True)
                X_val = X_val.join(encoded_val)

            encoded_test = pd.DataFrame(enc.transform(X_test[cat_cols]).toarray(),
                                        columns=enc.get_feature_names(cat_cols), index=X_test.index)
            X_test.drop(columns=cat_cols, inplace=True)
            X_test = X_test.join(encoded_test)

        ### Define Z
        Z_train = X_train[z_cols]
        if mode in ["train_val_test", "cv"]:
            Z_val = X_val[z_cols]
        Z_test = X_test[z_cols]

        X_train = X_train.drop(z_cols, axis=1)
        if mode in ["train_val_test", "cv"]:
            X_val = X_val.drop(z_cols, axis=1)
        X_test = X_test.drop(z_cols, axis=1)

        # Set datatytes
        return_dict["Z_train"+str_num] = Z_train.values.astype(np.int32)
        return_dict["X_train"+str_num] = X_train.astype(np.float32)
        if target=="categorical":
            return_dict["y_train" + str_num] = y_train.astype(np.int32).values.ravel()
        else:
            return_dict["y_train"+str_num] = y_train.astype(np.float32).values.ravel()

        if mode in ["train_val_test", "cv"]:
            return_dict["Z_val"+str_num] = Z_val.values.astype(np.int32)
            return_dict["X_val"+str_num] = X_val.astype(np.float32)
            if target == "categorical":
                return_dict["y_val" + str_num] = y_val.astype(np.int32).values.ravel()
            else:
                return_dict["y_val"+str_num] = y_val.astype(np.float32).values.ravel()

        return_dict["Z_test"+str_num] = Z_test.values.astype(np.int32)
        return_dict["X_test"+str_num] = X_test.astype(np.float32)
        if target=="categorical":
            return_dict["y_test" + str_num] = y_test.astype(np.int32).values.ravel()
        else:
            return_dict["y_test"+str_num] = y_test.astype(np.float32).values.ravel()


    with open(f"../data/prepared/{dataset_name}/{new_path}/data_dict.pickle", 'wb') as handle:
        pickle.dump(return_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)




