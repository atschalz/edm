import category_encoders as ce
import numpy as np
import pandas as pd
import gpboost as gpb
from utils.evaluation import get_metrics
from sklearn.metrics import log_loss
from tensorflow_addons.metrics import F1Score

from sklearn.model_selection import KFold, StratifiedKFold
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import roc_auc_score as auroc
from sklearn.metrics import f1_score as f1
from sklearn.linear_model import Lasso, LogisticRegression
import xgboost as xgb


def glmm5CV_encode_multiple_features_gpboost(Z_train, Z_val, Z_test, X_train, X_val, X_test, y_train, qs, RS):
    if np.unique(y_train).shape[0]==2:
        likelihood = "binary"
    else:
        likelihood = "gaussian"

    z_glmm_encoded_train = pd.DataFrame(Z_train, index=X_train.index)

    #     encoders = {fold: {q_num: ce.GLMMEncoder() for q_num in range(len(qs))} for fold in range(5)}
    kf_enc = KFold(n_splits=5, shuffle=True, random_state=RS)
    split_enc = kf_enc.split(Z_train, y_train)

    for num, (fit_indices, transform_indices) in enumerate(split_enc):
        Z_fit = Z_train[fit_indices]
        y_fit = y_train.iloc[fit_indices]
        Z_transform = Z_train[transform_indices]
        # y_transform = y_train.iloc[transform_indices]
        for q_num in range(len(qs)):
            print(f"Fit GLMM for fold {num} and feature {q_num}")
            gp_model = gpb.GPModel(group_data=Z_fit[:,[q_num]], likelihood=likelihood)
            gp_model.fit(y=y_fit, X=pd.DataFrame(np.ones([Z_fit.shape[0],1]),columns=["Intercept"]))

            n = Z_fit.shape[0]
            group = np.arange(n)
            m = qs[q_num]
            for i in range(m):
                group[int(i * n / m):int((i + 1) * n / m)] = i
            all_training_data_random_effects = gp_model.predict_training_data_random_effects()
            temp_mapping = dict(pd.concat([pd.DataFrame(Z_fit[:,[q_num]]),all_training_data_random_effects],axis=1).groupby(0).mean()["Group_1"])
            final_mapping = {i: float(gp_model.get_coef().values) if i not in list(temp_mapping.keys()) else temp_mapping[i] for i in range(qs[q_num])}

            z_glmm_encoded_train.iloc[transform_indices, q_num] = z_glmm_encoded_train.iloc[transform_indices, q_num].apply(lambda x: final_mapping[x])
    z_glmm_encoded_train = z_glmm_encoded_train.values

    encoded_val = []
    encoded_test = []
    for q_num in range(len(qs)):
        print(f"Fit GLMM on whole train data for feature {q_num}")
        gp_model = gpb.GPModel(group_data=Z_train[:,[q_num]], likelihood=likelihood)
        gp_model.fit(y=y_train, X=pd.DataFrame(np.ones([Z_train.shape[0],1]),columns=["Intercept"]))

        n = X_train.shape[0]
        group = np.arange(n)
        m = qs[q_num]
        for i in range(m):
            group[int(i * n / m):int((i + 1) * n / m)] = i
        all_training_data_random_effects = gp_model.predict_training_data_random_effects()
        temp_mapping = dict(pd.concat([pd.DataFrame(Z_train[:,[q_num]]),all_training_data_random_effects],axis=1).groupby(0).mean()["Group_1"])
        final_mapping = {i: float(gp_model.get_coef().values) if i not in list(temp_mapping.keys()) else temp_mapping[i] for i in range(qs[q_num])}

        encoded_val.append(pd.Series(Z_val[:,q_num]).apply(lambda x: final_mapping[x]))
        encoded_test.append(pd.Series(Z_test[:,q_num]).apply(lambda x: final_mapping[x]))

        z_glmm_encoded_val = pd.concat(encoded_val, axis=1).values
        z_glmm_encoded_test = pd.concat(encoded_test, axis=1).values

    return z_glmm_encoded_train, z_glmm_encoded_val, z_glmm_encoded_test


def glmm5CV_encode_multiple_features_statsmodels(Z_train, Z_val, Z_test, X_train, X_val, X_test, y_train, qs, RS):
    Z_train_df = pd.DataFrame(Z_train, index=X_train.index)
    Z_val_df = pd.DataFrame(Z_val, index=X_val.index)
    Z_test_df = pd.DataFrame(Z_test, index=X_test.index)

    z_glmm_encoded_train = pd.DataFrame(np.zeros(Z_train_df.shape), index=X_train.index)

    encoders = {fold: {q_num: ce.GLMMEncoder() for q_num in range(len(qs))} for fold in range(5)}
    kf_enc = KFold(n_splits=5, shuffle=True, random_state=RS)
    split_enc = kf_enc.split(Z_train_df, y_train)

    for num, (fit_indices, transform_indices) in enumerate(split_enc):
        Z_fit = Z_train_df.iloc[fit_indices]
        y_fit = y_train.iloc[fit_indices]
        Z_transform = Z_train_df.iloc[transform_indices]
        # y_transform = y_train.iloc[transform_indices]
        for q_num in range(len(qs)):
            print(f"Fit GLMM for fold {num} and feature {q_num}")
            encoders[num][q_num].fit(Z_fit.astype(object)[[q_num]], y_fit)
            #         encoded.append(encoder.transform(pd.DataFrame(Z_train,index=X_train.index).astype(object)[[q_num]]))
            z_glmm_encoded_train.iloc[transform_indices, q_num] = encoders[num][q_num].transform(Z_transform[[q_num]])[
                q_num]
    z_glmm_encoded_train = z_glmm_encoded_train.values

    encoded_val = []
    encoded_test = []
    for q_num in range(len(qs)):
        print(f"Fit GLMM on whole train data for feature {q_num}")
        encoder = ce.GLMMEncoder()
        encoder.fit(Z_train_df.astype(object)[[q_num]], y_train)

        encoded_val.append(encoder.transform(Z_val_df.astype(object)[[q_num]]))
        encoded_test.append(encoder.transform(Z_test_df.astype(object)[[q_num]]))

        z_glmm_encoded_val = pd.concat(encoded_val, axis=1).values
        z_glmm_encoded_test = pd.concat(encoded_test, axis=1).values

    return z_glmm_encoded_train, z_glmm_encoded_val, z_glmm_encoded_test


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

def tune_xgboost(X, y, X_test, y_test, target, max_evals=50, early_stopping_rounds=10,seed=0):
    '''Algorithm to iteratively tune XGBoost in 4 steps using Bayesian optimization
    Search ranges:
        n_estimators = Uniform in [50, 500], int, (default = 100)
        learning_rate = Uniform in [0.001,0.5], float, (default=0.3)
        max_depth = Q-uniform in [1,18], int, (default = 6)
        min_child_weight = Q-uniform in [0,10], int, (default = 1)
        colsample_bytree = Uniform in [0.5,1], float, (default=1)
        subsample = Uniform in [0.5,1], float, (default=1)
        gamma: Uniform in [10^{-8},9], float, (default = 0)
        reg_alpha = Q-uniform in [10^{-8},10], int, (default = 0)
        reg_lambda = Uniform in [1,4], float, (default = 1)

    space['max_depth'] = hp.quniform("max_depth", 3, 18, 1)
    space['min_child_weight'] = hp.quniform('min_child_weight', 0, 10, 1)
    'n_estimators': hp.uniformint("n_estimators", 50, 500),
    'learning_rate': hp.uniform("learning_rate", 0.001, 0.2),



    '''
    # 1. Find optimal learning rate and no. of estimators
    if target == "continuous":
        xgb_model = xgb.XGBRegressor
        xgb_metric = "rmse"
        eval_metric = mse
        xgb_objective = "reg:squarederror"
    elif target == "binary":
        xgb_model = xgb.XGBClassifier
        xgb_metric = "auc"
        # eval_metric = lambda y_true, y_pred: -auroc(y_true, y_pred)
        eval_metric = log_loss
        xgb_objective = "binary:logistic"

    elif target == "categorical":
        nb_classes = np.unique(y).shape[0]
        xgb_model = xgb.XGBClassifier
        xgb_metric = "auc"
        # eval_metric = lambda y_true, y_pred: -auroc(y_true, y_pred, multi_class="ovo", average="macro")
        # eval_metric = lambda y_true, y_pred: -f1(y_true, y_pred, average="macro")
        eval_metric = lambda y_true, y_pred: -F1Score(num_classes=nb_classes, average="macro")(get_one_hot(y_true, nb_classes), y_pred).numpy()
        # eval_metric = log_loss
        xgb_objective = "multi:softproba"

    model = xgb_model(
        objective = xgb_objective,
        # eval_metric=xgb_metric,
        # early_stopping_rounds=early_stopping_rounds,
        seed=seed)
    model.fit(X, y,
              verbose=False)
    if target=="continuous":
        test_pred = model.predict(X_test)
    else:
        test_pred = model.predict_proba(X_test)
    print(f"Default performance on Test: {eval_metric(y_test, test_pred)}")

    if target == "continuous":
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    else:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    split = kf.split(X, y)
    X_train_list = []
    y_train_list = []
    X_val_list = []
    y_val_list = []
    for num, (train_indices, test_indices) in enumerate(split):
        X_train_list.append(X.iloc[train_indices])
        y_train_list.append(y[train_indices])
        X_val_list.append(X.iloc[test_indices])
        y_val_list.append(y[test_indices])


    space = {
        'n_estimators': hp.uniformint("n_estimators", 50, 500),
        'learning_rate': hp.uniform("learning_rate", 0.001, 0.5),
        'seed': seed
    }

    def objective(space):
        score_list = []
        for i in range(5):
            X_train = X_train_list[i]
            y_train = y_train_list[i]
            X_val = X_val_list[i]
            y_val = y_val_list[i]

            model = xgb_model(
                objective=xgb_objective,
                eval_metric=xgb_metric,
                early_stopping_rounds=early_stopping_rounds,
                n_estimators=space['n_estimators'],
                learning_rate=space['learning_rate'],
                seed = space['seed']
            )

            evaluation = [(X_train, y_train), (X_val, y_val)]

            model.fit(X_train, y_train,
                      eval_set=evaluation,
                      verbose=False)

            if target=="continuous":
                pred = model.predict(X_val)
            else:
                pred = model.predict_proba(X_val)
            score_list.append(eval_metric(y_val, pred))
        score = np.mean(score_list)
        print(f"SCORE: {score}")
        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()

    best_hyperparams = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=max_evals,
                            trials=trials)

    final_hyperparameters = best_hyperparams

    print("The best hyperparameters after step 1  are : ", "\n")
    print(final_hyperparameters)



    # 2. Tune max_depth and min_child_weight
    space = final_hyperparameters
    space['seed'] = seed
    space['max_depth'] = hp.quniform("max_depth", 1, 18, 1)
    space['min_child_weight'] = hp.quniform('min_child_weight', 0, 10, 1)

    model = xgb_model(
        objective = xgb_objective,
        # eval_metric=xgb_metric,
        # early_stopping_rounds=early_stopping_rounds,
        n_estimators=int(space['n_estimators']),
        learning_rate=space['learning_rate'],
        seed=seed)
    model.fit(X, y,
              verbose=False)
    if target=="continuous":
        test_pred = model.predict(X_test)
    else:
        test_pred = model.predict_proba(X_test)
    print(f"Test Performance after first tuning round: {eval_metric(y_test, test_pred)}")


    def objective(space):
        score_list = []
        for i in range(5):
            X_train = X_train_list[i]
            y_train = y_train_list[i]
            X_val = X_val_list[i]
            y_val = y_val_list[i]

            model = xgb_model(
                objective=xgb_objective,
                eval_metric=xgb_metric,
                early_stopping_rounds=early_stopping_rounds,
                n_estimators=int(space['n_estimators']),
                learning_rate=space['learning_rate'],
                max_depth=int(space['max_depth']),
                min_child_weight=int(space['min_child_weight']),
                seed=space['seed']
            )

            evaluation = [(X_train, y_train), (X_val, y_val)]

            model.fit(X_train, y_train,
                      eval_set=evaluation,
                      verbose=False)

            if target=="continuous":
                pred = model.predict(X_val)
            else:
                pred = model.predict_proba(X_val)
            score_list.append(eval_metric(y_val, pred))
        score = np.mean(score_list)
        print(f"SCORE: {score}")
        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()

    best_hyperparams = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=max_evals,
                            trials=trials)

    for key in best_hyperparams:
        final_hyperparameters[key] = best_hyperparams[key]

    print("The best hyperparameters after step 2 are : ", "\n")
    print(final_hyperparameters)

    # 3. Tune colsample_bytree and sampling
    space = final_hyperparameters
    space['seed'] = seed
    space['colsample_bytree'] = hp.uniform('colsample_bytree', 0.5, 1)
    space['subsample'] = hp.uniform('subsample', 0.5, 1)

    model = xgb_model(
        objective = xgb_objective,
        # eval_metric=xgb_metric,
        # early_stopping_rounds=early_stopping_rounds,
        n_estimators=int(space['n_estimators']),
        learning_rate=space['learning_rate'],
        max_depth=int(space['max_depth']),
        min_child_weight=int(space['min_child_weight']),
        seed=seed)
    model.fit(X, y,
              verbose=False)
    if target=="continuous":
        test_pred = model.predict(X_test)
    else:
        test_pred = model.predict_proba(X_test)
    print(f"Test Performance after second tuning round: {eval_metric(y_test, test_pred)}")

    def objective(space):
        score_list = []
        for i in range(5):
            X_train = X_train_list[i]
            y_train = y_train_list[i]
            X_val = X_val_list[i]
            y_val = y_val_list[i]

            model = xgb_model(
                objective=xgb_objective,
                eval_metric=xgb_metric,
                early_stopping_rounds=early_stopping_rounds,
                n_estimators=int(space['n_estimators']),
                learning_rate=space['learning_rate'],
                max_depth=int(space['max_depth']),
                min_child_weight=int(space['min_child_weight']),
                colsample_bytree=space['colsample_bytree'],
                subsample=space['subsample'],
                seed=space['seed']
            )

            evaluation = [(X_train, y_train), (X_val, y_val)]

            model.fit(X_train, y_train,
                      eval_set=evaluation,
                      verbose=False)

            if target=="continuous":
                pred = model.predict(X_val)
            else:
                pred = model.predict_proba(X_val)
            score_list.append(eval_metric(y_val, pred))
        score = np.mean(score_list)
        print(f"SCORE: {score}")
        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()

    best_hyperparams = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=max_evals,
                            trials=trials)

    for key in best_hyperparams:
        final_hyperparameters[key] = best_hyperparams[key]

    print("The best hyperparameters after step 3 are : ", "\n")
    print(final_hyperparameters)

    # 4. Tune alpha,lambda & gamma
    space = final_hyperparameters
    space['seed'] = seed
    space['gamma'] = hp.uniform('gamma', 1e-8, 9)
    space['reg_alpha'] = hp.quniform('reg_alpha', 1e-8, 10, 1)
    space['reg_lambda'] = hp.uniform('reg_lambda', 1, 4)

    model = xgb_model(
        objective = xgb_objective,
        # eval_metric=xgb_metric,
        # early_stopping_rounds=early_stopping_rounds,
        n_estimators=int(space['n_estimators']),
        learning_rate=space['learning_rate'],
        max_depth=int(space['max_depth']),
        min_child_weight=int(space['min_child_weight']),
        colsample_bytree=space['colsample_bytree'],
        subsample=space['subsample'],
        seed=seed)
    model.fit(X, y,
              verbose=False)
    if target=="continuous":
        test_pred = model.predict(X_test)
    else:
        test_pred = model.predict_proba(X_test)
    print(f"Test Performance after third tuning round: {eval_metric(y_test, test_pred)}")


    def objective(space):
        score_list = []
        for i in range(5):
            X_train = X_train_list[i]
            y_train = y_train_list[i]
            X_val = X_val_list[i]
            y_val = y_val_list[i]

            model = xgb_model(
                objective=xgb_objective,
                eval_metric=xgb_metric,
                early_stopping_rounds=early_stopping_rounds,
                n_estimators=int(space['n_estimators']),
                learning_rate=space['learning_rate'],
                max_depth=int(space['max_depth']),
                min_child_weight=int(space['min_child_weight']),
                colsample_bytree=space['colsample_bytree'],
                subsample=space['subsample'],
                gamma=space['gamma'],
                reg_alpha=int(space['reg_alpha']),
                reg_lambda=space['reg_lambda'],
                seed=space['seed']
            )

            evaluation = [(X_train, y_train), (X_val, y_val)]

            model.fit(X_train, y_train,
                      eval_set=evaluation,
                      verbose=False)

            if target=="continuous":
                pred = model.predict(X_val)
            else:
                pred = model.predict_proba(X_val)
            score_list.append(eval_metric(y_val, pred))
        score = np.mean(score_list)
        print(f"SCORE: {score}")
        return {'loss': score, 'status': STATUS_OK}


    trials = Trials()

    best_hyperparams = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=max_evals,
                            trials=trials)

    for key in best_hyperparams:
        final_hyperparameters[key] = best_hyperparams[key]

    print("The best hyperparameters are : ", "\n")
    print(final_hyperparameters)

    model = xgb_model(
        objective = xgb_objective,
        # eval_metric=xgb_metric,
        # early_stopping_rounds=early_stopping_rounds,
        n_estimators=int(final_hyperparameters['n_estimators']),
        learning_rate=final_hyperparameters['learning_rate'],
        max_depth=int(final_hyperparameters['max_depth']),
        min_child_weight=int(final_hyperparameters['min_child_weight']),
        colsample_bytree=final_hyperparameters['colsample_bytree'],
        subsample=final_hyperparameters['subsample'],
        gamma=final_hyperparameters['gamma'],
        reg_alpha=int(final_hyperparameters['reg_alpha']),
        reg_lambda=final_hyperparameters['reg_lambda'],
        seed=seed)
    model.fit(X, y,
              verbose=False)
    if target=="continuous":
        test_pred = model.predict(X_test)
    else:
        test_pred = model.predict_proba(X_test)
    print(f"Test Performance after last tuning round: {eval_metric(y_test, test_pred)}")

    return final_hyperparameters

def evaluate_xgb(X_train, y_train, X_test, y_test, target, tune=False, max_evals=20, early_stopping_rounds=10, seed=0, target_scaler=None):
    results = {}
    if target =="continuous":
        xgb_model_type = xgb.XGBRegressor
        xgb_objective = "reg:squarederror"
    elif target =="binary":
        xgb_model_type = xgb.XGBClassifier
        xgb_objective = "binary:logistic"
    elif target =="categorical":
        xgb_model_type = xgb.XGBClassifier
        nb_classes = np.unique(y_train).shape[0]
        xgb_objective = "multi:softproba"


    if tune:
        final_hyperparameters = tune_xgboost(X_train, y_train, X_test, y_test, target, max_evals=max_evals,early_stopping_rounds=early_stopping_rounds,seed=seed)
        xgb_model = xgb_model_type(
            objective = xgb_objective,
            # eval_metric=xgb_metric,
            # early_stopping_rounds=early_stopping_rounds,
            n_estimators=int(final_hyperparameters['n_estimators']),
            learning_rate=final_hyperparameters['learning_rate'],
            max_depth=int(final_hyperparameters['max_depth']),
            min_child_weight=int(final_hyperparameters['min_child_weight']),
            colsample_bytree=final_hyperparameters['colsample_bytree'],
            subsample=final_hyperparameters['subsample'],
            gamma=final_hyperparameters['gamma'],
            reg_alpha=int(final_hyperparameters['reg_alpha']),
            reg_lambda=final_hyperparameters['reg_lambda'],
            seed=seed)

    else:
        xgb_model = xgb_model_type(
            objective = xgb_objective,
            seed=seed)

    xgb_model.fit(X_train, y_train)

    if target == "continuous":
        y_train_pred = xgb_model.predict(X_train)
        y_test_pred = xgb_model.predict(X_test)

        y_train_rescaled = target_scaler.inverse_transform(y_train.reshape(-1,1)).ravel()
        y_train_pred_rescaled = target_scaler.inverse_transform(y_train_pred.reshape(-1,1)).ravel()
        y_test_rescaled= target_scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
        y_test_pred_rescaled = target_scaler.inverse_transform(y_test_pred.reshape(-1,1)).ravel()

        eval_res_train = get_metrics(y_train_rescaled, y_train_pred_rescaled, target=target)
        eval_res_test = get_metrics(y_test_rescaled, y_test_pred_rescaled, target=target)
    else:
        y_train_pred = xgb_model.predict_proba(X_train)
        y_test_pred = xgb_model.predict_proba(X_test)
        if target == "binary":
            eval_res_train = get_metrics(y_train, y_train_pred[:,1], target=target)
            eval_res_test = get_metrics(y_test, y_test_pred[:,1], target=target)
        else:
            eval_res_train = get_metrics(get_one_hot(y_train, nb_classes), y_train_pred, target=target)
            eval_res_test = get_metrics(get_one_hot(y_test, nb_classes), y_test_pred, target=target)

    for metric in eval_res_train.keys():
        results[metric + " Train"] = eval_res_train[metric]
    for metric in eval_res_test.keys():
        results[metric + " Test"] = eval_res_test[metric]

    xgb_feat = pd.DataFrame([xgb_model.feature_importances_], columns=X_train.columns).transpose()

    return results, xgb_feat

def tune_lasso(X, y, max_evals=20, seed=0):
    '''Algorithm to tune alpha for Lasso Regression'''

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    split = kf.split(X, y)
    X_train_list = []
    y_train_list = []
    X_val_list = []
    y_val_list = []
    for num, (train_indices, test_indices) in enumerate(split):
        X_train_list.append(X.iloc[train_indices])
        y_train_list.append(y[train_indices])
        X_val_list.append(X.iloc[test_indices])
        y_val_list.append(y[test_indices])


    space = {
        'alpha': hp.uniform("alpha", 1e-10,0.5),
        'seed': seed
    }

    def objective(space):
        score_list = []
        for i in range(5):
            X_train = X_train_list[i]
            y_train = y_train_list[i]
            X_val = X_val_list[i]
            y_val = y_val_list[i]

            model = Lasso(alpha=space["alpha"],
                          random_state=space['seed']
                          )

            model.fit(X_train, y_train)

            # if target=="continuous":
            pred = model.predict(X_val)
            # else:
            #     pred = model.predict_proba(X_val)
            score_list.append(mse(y_val, pred))
        score = np.mean(score_list)
        print(f"SCORE: {score}")
        return {'loss': score, 'status': STATUS_OK}


    trials = Trials()

    best_hyperparams = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=max_evals,
                            trials=trials)

    final_hyperparameters = best_hyperparams

    print("The best hyperparameters are : ", "\n")
    print(final_hyperparameters)

    return final_hyperparameters


def evaluate_lr(X_train, y_train, X_test, y_test, target, tune=False, max_evals=20, seed=0, target_scaler=None):
    results = {}

    if tune:
        final_hyperparameters = tune_lasso(X_train, y_train, max_evals=max_evals, seed=seed)
        lr = Lasso(alpha=final_hyperparameters["alpha"],
                   random_state=seed)

    else:
        lr = Lasso(random_state=seed, alpha=0.001)
    lr.fit(X_train, y_train)

    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)

    y_train_rescaled = target_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_train_pred_rescaled = target_scaler.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()
    y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_test_pred_rescaled = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()

    eval_res_train = get_metrics(y_train_rescaled, y_train_pred_rescaled, target=target)
    eval_res_test = get_metrics(y_test_rescaled, y_test_pred_rescaled, target=target)

    eval_res_train = get_metrics(y_train_rescaled, y_train_pred_rescaled, target=target)
    for metric in eval_res_train.keys():
        results[metric + " Train"] = eval_res_train[metric]
    eval_res_test = get_metrics(y_test_rescaled, y_test_pred_rescaled, target=target)
    for metric in eval_res_test.keys():
        results[metric + " Test"] = eval_res_test[metric]

    lr_feat = pd.DataFrame([lr.coef_], columns=X_train.columns).transpose()

    return results, lr_feat


def evaluate_logreg(X_train, y_train, X_test, y_test, target, tune=False, max_evals=20, seed=0):
    results = {}
    nb_classes = np.unique(y_train).shape[0]
    if tune:
        final_hyperparameters = tune_logreg_multiclass(X_train, y_train, max_evals=max_evals, seed=seed)
        lr = LogisticRegression(penalty="l2",
                                       solver="lbfgs",
                                       C=final_hyperparameters["C"],
                                       max_iter=10000,
                                       random_state=seed
                                       )

    else:
        lr = LogisticRegression(penalty="l2",
                                       solver="lbfgs",
                                       max_iter=10000,
                                       random_state=seed
                                       )
    lr.fit(X_train.values, y_train)

    y_train_pred_lr = lr.predict_proba(X_train.values)
    y_test_pred_lr = lr.predict_proba(X_test.values)

    if target == "binary":
        eval_res_train = get_metrics(y_train, y_train_pred_lr[:,1], target=target)
        eval_res_test = get_metrics(y_test, y_test_pred_lr[:,1], target=target)
    else:
        eval_res_train = get_metrics(get_one_hot(y_train, nb_classes), y_train_pred_lr, target=target)
        eval_res_test = get_metrics(get_one_hot(y_test, nb_classes), y_test_pred_lr, target=target)
    for metric in eval_res_train.keys():
        results[metric + " Train"] = eval_res_train[metric]
    for metric in eval_res_test.keys():
        results[metric + " Test"] = eval_res_test[metric]

    lr_feat = pd.DataFrame(lr.coef_, columns=X_train.columns).transpose()

    return results, lr_feat


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def tune_logreg_multiclass(X, y, max_evals=20, seed=0):
    '''Algorithm to tune C for Logistic Regression'''

    n_classes = np.unique(y).shape[0]

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    split = kf.split(X, y)
    X_train_list = []
    y_train_list = []
    X_val_list = []
    y_val_list = []
    for num, (train_indices, test_indices) in enumerate(split):
        X_train_list.append(X.iloc[train_indices])
        y_train_list.append(y[train_indices])
        X_val_list.append(X.iloc[test_indices])
        y_val_list.append(y[test_indices])


    space = {
        'C': hp.uniform("C", 1e-10, 1.0),
        'seed': seed
    }

    def objective(space):
        score_list = []
        for i in range(5):
            X_train = X_train_list[i]
            y_train = y_train_list[i]
            X_val = X_val_list[i]
            y_val = y_val_list[i]

            model = LogisticRegression(penalty="l2",
                                       solver="lbfgs",
                                       C=space["C"],
                                       max_iter=10000,
                                       random_state=space['seed']
                                       )

            model.fit(X_train.values, y_train)

            # if target=="continuous":
            pred = model.predict_proba(X_val.values)
            # else:
            #     pred = model.predict_proba(X_val)
            score_list.append(log_loss(y_val, pred))
        score = np.mean(score_list)
        print(f"SCORE: {score}")
        return {'loss': score, 'status': STATUS_OK}


    trials = Trials()

    best_hyperparams = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=max_evals,
                            trials=trials)

    final_hyperparameters = best_hyperparams

    print("The best hyperparameters are : ", "\n")
    print(final_hyperparameters)

    return final_hyperparameters


def tune_logreg_binary(X, y, target, max_evals=50, seed=0):
    '''Algorithm to tune C for Logistic Regression'''

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    split = kf.split(X, y)
    X_train_list = []
    y_train_list = []
    X_val_list = []
    y_val_list = []
    for num, (train_indices, test_indices) in enumerate(split):
        X_train_list.append(X.iloc[train_indices])
        y_train_list.append(y[train_indices])
        X_val_list.append(X.iloc[test_indices])
        y_val_list.append(y[test_indices])


    space = {
        'C': hp.uniform("C", 1e-10, 1.0),
        'seed': seed
    }

    def objective(space):
        score_list = []
        for i in range(5):
            X_train = X_train_list[i]
            y_train = y_train_list[i]
            X_val = X_val_list[i]
            y_val = y_val_list[i]

            model = LogisticRegression(penalty="l2",
                                       solver="lbfgs",
                                       C=space["C"],
                                       max_iter=10000,
                                       random_state=space['seed']
                                       )

            model.fit(X_train.values, y_train)

            # if target=="continuous":
            pred = model.predict_proba(X_val.values)
            # else:
            #     pred = model.predict_proba(X_val)
            score_list.append(log_loss(y_val, pred))
        score = np.mean(score_list)
        print(f"SCORE: {score}")
        return {'loss': score, 'status': STATUS_OK}


    trials = Trials()

    best_hyperparams = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=max_evals,
                            trials=trials)

    final_hyperparameters = best_hyperparams

    print("The best hyperparameters are : ", "\n")
    print(final_hyperparameters)

    return final_hyperparameters