import os
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.fe_models import get_model
from utils.evaluation import get_metrics
from lmmnn.utils import generate_data
from lmmnn.nn import reg_nn_lmm

from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Reshape, Embedding, Concatenate
from tensorflow.keras.activations import sigmoid

from vis.utils.utils import apply_modifications

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import roc_auc_score as auroc
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from category_encoders import TargetEncoder
from tensorflow_addons.metrics import F1Score, RSquare
import time

import pickle

# helper function
def update_layer_activation(model, activation, index=-1):
    model.layers[index].activation = activation
    return apply_modifications(model)

def train_models(train_data, val_data, test_data, config, RS=42, save_results=True, save_path=""):
    tf.random.set_seed(RS)
    # Prepare results
    results = {}

    # Check whether all directories are created
    if save_results:
        save_path_split = save_path.split("/")
        for i in range(len(save_path_split)):
            if not os.path.exists("/".join(save_path_split[:i + 1])):
                os.mkdir("/".join(save_path_split[:i + 1]))


    # Get data
    X_train, Z_train, y_train, z_ohe_encoded_train, z_target_encoded_train = train_data
    X_val, Z_val, y_val, z_ohe_encoded_val, z_target_encoded_val = val_data
    X_test, Z_test, y_test, z_ohe_encoded_test, z_target_encoded_test = test_data

    # Load parameters from config
    # General parameters
    target = config["general_parameters"]["target"]
    if target=="binary":
        activation = "sigmoid"
        activation_layer = tf.keras.activations.sigmoid
        loss = tf.keras.losses.BinaryCrossentropy()
        xgb = XGBClassifier
        lr = LogisticRegression(solver='lbfgs', max_iter=10000)
        num_outputs = 1
    elif target=="continuous":
        activation = "linear"
        activation_layer = tf.keras.activations.linear
        loss = tf.keras.losses.MeanSquaredError()
        xgb = XGBRegressor
        lr = LinearRegression()
        num_outputs = 1
    elif target == "categorical":
        activation = "softmax"
        activation_layer = tf.keras.activations.softmax
        loss = tf.keras.losses.CategoricalCrossentropy()
        xgb = XGBClassifier
        lr = LogisticRegression(solver='lbfgs', max_iter=10000)
        num_outputs = np.unique(y_train).shape[0]

    if target == "categorical":
        y_train_nn = tf.one_hot(y_train,num_outputs)
        y_val_nn = tf.one_hot(y_val,num_outputs)
        y_test_nn = tf.one_hot(y_test,num_outputs)
    elif target in ["continuous", "binary"]:
        y_train_nn = y_train
        y_val_nn = y_val
        y_test_nn = y_test


    metrics = config["general_parameters"]["metrics"]
    model_name = config["general_parameters"]["model_name"]


    # NN parameters
    epochs = config["nn_parameters"]["epochs"]
    batch_size = config["nn_parameters"]["batch_size"]
    patience = config["nn_parameters"]["patience"]
    stop_metric = config["nn_parameters"]["stop_metric"]

    # AutoGluon specific parameters

    # Infer necessary parameters
    qs = [np.unique(i).shape[0] for i in Z_train.transpose()]
    d = X_train.shape[1] # columns
    n = X_train.shape[0] # rows
    perc_numeric = d/(d+Z_train.shape[1])

    # Define base model
    print(f"Load base model")
    ### Lukas: changed get_model call with new parameter output_size
    base_model, optimizer = get_model(model_name=model_name, input_size=d, output_size=num_outputs, target=target, perc_numeric=perc_numeric, RS=RS)

    print(f"Train XGBoost without z features")
    start = time.time()
    if target=="categorical":
        xgb_metric = "merror"
    if target in ["binary", "continuous"]:
        xgb_metric = "error"

    if X_train.shape[1 ]!=0:
        results["XGB"] = {}
        xgb_model = xgb(eval_metric=xgb_metric)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val ,y_val)], verbose=False)
        end = time.time()
        if target == "binary":
            y_train_pred_xgb = xgb_model.predict_proba(X_train)[:,1]
            y_test_pred_xgb = xgb_model.predict_proba(X_test)[:,1]
        elif target == "categorical":
            y_train_pred_xgb = xgb_model.predict_proba(X_train)
            y_test_pred_xgb = xgb_model.predict_proba(X_test)
        elif target=="continuous":
            y_train_pred_xgb = xgb_model.predict(X_train)
            y_test_pred_xgb = xgb_model.predict(X_test)

        eval_res_train = get_metrics(y_train_nn, y_train_pred_xgb, target=target)
        for metric in eval_res_train.keys():
            results["XGB"][metric + " Train"] = eval_res_train[metric]
        eval_res_test = get_metrics(y_test_nn, y_test_pred_xgb, target=target)
        for metric in eval_res_test.keys():
            results["XGB"][metric + " Test"] = eval_res_test[metric]
        results["XGB"]["Time"] = round(end - start, 2)

    else:
        eval_res_train = get_metrics(y_train, np.zeros(y_train.shape[0]), target=target)
        for metric in eval_res_train.keys():
            results["XGB"][metric + " Train"] = eval_res_train[metric]
        eval_res_test = get_metrics(y_test, np.zeros(y_test.shape[0]), target=target)
        for metric in eval_res_test.keys():
            results["XGB"][metric + " Test"] = eval_res_test[metric]
        results["XGB"]["Time"] = 0


    print(f"Train XGBoost with target encoding")
    results["XGB_te"] = {}
    start = time.time()
    xgb_model_te = xgb(eval_metric=xgb_metric)
    xgb_model_te.fit(np.append(X_train ,z_target_encoded_train ,axis=1), y_train, eval_set=[(np.append(X_val ,z_target_encoded_val ,axis=1), y_val)], verbose=False)
    end = time.time()
    if target == "binary":
        y_train_pred_xgb_te = xgb_model_te.predict_proba(np.append(X_train ,z_target_encoded_train ,axis=1))[:, 1]
        y_test_pred_xgb_te = xgb_model_te.predict_proba(np.append(X_test ,z_target_encoded_test ,axis=1))[:, 1]
    elif target == "categorical":
        y_train_pred_xgb_te = xgb_model_te.predict_proba(np.append(X_train ,z_target_encoded_train ,axis=1))
        y_test_pred_xgb_te = xgb_model_te.predict_proba(np.append(X_test ,z_target_encoded_test ,axis=1))
    elif target == "continuous":
        y_train_pred_xgb_te = xgb_model_te.predict(np.append(X_train ,z_target_encoded_train ,axis=1))
        y_test_pred_xgb_te = xgb_model_te.predict(np.append(X_test ,z_target_encoded_test ,axis=1))

    eval_res_train = get_metrics(y_train_nn, y_train_pred_xgb_te, target=target)
    for metric in eval_res_train.keys():
        results["XGB_te"][metric + " Train"] = eval_res_train[metric]
    eval_res_test = get_metrics(y_test_nn, y_test_pred_xgb_te, target=target)
    for metric in eval_res_test.keys():
        results["XGB_te"][metric + " Test"] = eval_res_test[metric]
    results["XGB_te"]["Time"] = round(end - start, 2)

    print(f"Train Linear Model without z features")
    start = time.time()
    if X_train.shape[1 ]!=0:
        results["LR"] = {}
        lr_model = lr
        lr_model.fit(X_train, y_train)
        end = time.time()
        if target == "binary":
            y_train_pred_lr = lr_model.predict_proba(X_train)[:,1]
            y_test_pred_lr = lr_model.predict_proba(X_test)[:,1]
        elif target == "categorical":
            y_train_pred_lr = lr_model.predict_proba(X_train)
            y_test_pred_lr = lr_model.predict_proba(X_test)
        elif target=="continuous":
            y_train_pred_lr = lr_model.predict(X_train)
            y_test_pred_lr = lr_model.predict(X_test)

        eval_res_train = get_metrics(y_train_nn, y_train_pred_lr, target=target)
        for metric in eval_res_train.keys():
            results["LR"][metric + " Train"] = eval_res_train[metric]
        eval_res_test = get_metrics(y_test_nn, y_test_pred_lr, target=target)
        for metric in eval_res_test.keys():
            results["LR"][metric + " Test"] = eval_res_test[metric]
        results["LR"]["Time"] = round(end - start, 2)

    else:
        eval_res_train = get_metrics(y_train, np.zeros(y_train.shape[0]), target=target)
        for metric in eval_res_train.keys():
            results["LR"][metric + " Train"] = eval_res_train[metric]
        eval_res_test = get_metrics(y_test, np.zeros(y_test.shape[0]), target=target)
        for metric in eval_res_test.keys():
            results["LR"][metric + " Test"] = eval_res_test[metric]
        results["LR"]["Time"] = 0


    print(f"Train Linear Model with target encoding")
    results["LR_te"] = {}
    start = time.time()
    lr_model_te = lr
    lr_model_te.fit(np.append(X_train ,z_target_encoded_train ,axis=1), y_train)
    end = time.time()
    if target == "binary":
        y_train_pred_lr_te = lr_model_te.predict_proba(np.append(X_train ,z_target_encoded_train ,axis=1))[:, 1]
        y_test_pred_lr_te = lr_model_te.predict_proba(np.append(X_test ,z_target_encoded_test ,axis=1))[:, 1]
    elif target == "categorical":
        y_train_pred_lr_te = lr_model_te.predict_proba(np.append(X_train ,z_target_encoded_train ,axis=1))
        y_test_pred_lr_te = lr_model_te.predict_proba(np.append(X_test ,z_target_encoded_test ,axis=1))
    elif target == "continuous":
        y_train_pred_lr_te = lr_model_te.predict(np.append(X_train ,z_target_encoded_train ,axis=1))
        y_test_pred_lr_te = lr_model_te.predict(np.append(X_test ,z_target_encoded_test ,axis=1))

    eval_res_train = get_metrics(y_train_nn, y_train_pred_lr_te, target=target)
    for metric in eval_res_train.keys():
        results["LR_te"][metric + " Train"] = eval_res_train[metric]
    eval_res_test = get_metrics(y_test_nn, y_test_pred_lr_te, target=target)
    for metric in eval_res_test.keys():
        results["LR_te"][metric + " Test"] = eval_res_test[metric]
    results["LR_te"]["Time"] = round(end - start, 2)


    metrics_use = []
    if "auc" in metrics:
        metrics_use.append(tf.keras.metrics.AUC(from_logits=True, name="auc"))
    if "accuracy" in metrics:
        if target =="binary":
            metrics_use.append(tf.keras.metrics.Accuracy(name="accuracy"))
        elif target == "categorical":
            metrics_use.append(tf.keras.metrics.CategoricalAccuracy(name="accuracy"))
    if "f1" in metrics:
        if target == "binary":
            metrics_use.append(F1Score(num_classes=2, average="micro", name="f1"))
        elif target == "categorical":
            metrics_use.append(F1Score(num_classes=num_outputs, average="weighted", name="f1"))
    if "r2" in metrics:
        metrics_use.append(RSquare(name="r2"))
    if "mse" in metrics:
        metrics_use.append(tf.keras.metrics.MeanSquaredError(name="mse"))
    if stop_metric in ["auc", "accuracy", "f1", "r2", "val_auc", "val_accuracy", "val_f1", "val_r2"]:
        stop_mode = "max"
    else:
        stop_mode = "min"

    print(f"Train NN without Z features")
    results["NN"] = {}
    start = time.time()
    if X_train.shape[1 ]!=0:
        model_nn = tf.keras.models.clone_model(base_model)
        ### Lukas: change activation function in output layer
        model_nn.build((n,d))
        update_layer_activation(model=model_nn, activation=activation_layer)
        # model_nn.add(Dense(num_outputs, activation=activation))
        model_nn.compile(loss=loss, optimizer=optimizer, metrics = metrics_use)
        callback = tf.keras.callbacks.EarlyStopping(monitor=stop_metric, patience=patience, mode=stop_mode)
        model_nn.fit(X_train, y_train_nn,
                     validation_data= [X_val, y_val_nn],
                     epochs=epochs, batch_size=batch_size, callbacks=[callback])
        end = time.time()

        y_train_pred_nn = model_nn.predict(X_train ,batch_size=batch_size)
        y_test_pred_nn = model_nn.predict(X_test ,batch_size=batch_size)

        eval_res_train = get_metrics(y_train_nn, y_train_pred_nn, target=target)
        for metric in eval_res_train.keys():
            results["NN"][metric + " Train"] = eval_res_train[metric]
        eval_res_test = get_metrics(y_test_nn, y_test_pred_nn, target=target)
        for metric in eval_res_test.keys():
            results["NN"][metric + " Test"] = eval_res_test[metric]
        results["NN"]["Time"] = round(end - start, 2)

        del model_nn
    else:
        eval_res_train = get_metrics(y_train_nn, np.zeros(y_train_nn.shape[0]), target=target)
        for metric in eval_res_train.keys():
            results["NN"][metric + " Train"] = eval_res_train[metric]
        eval_res_test = get_metrics(y_test_nn, np.zeros(y_test_nn.shape[0]), target=target)
        for metric in eval_res_test.keys():
            results["NN"][metric + " Test"] = eval_res_test[metric]
        results["NN"]["Time"] = 0

    print(f"Train NN with target encoding")
    results["NN_te"] = {}
    start = time.time()
    # model_te = tf.keras.models.clone_model(base_model)
    model_te, optimizer = get_model(model_name=model_name, input_size=d+z_target_encoded_train.shape[1], output_size=num_outputs, RS=RS)
    ### Lukas: change activation function in output layer
    model_te.build((n,d+z_target_encoded_train.shape[1]))
    update_layer_activation(model=model_te, activation=activation_layer)
    # model_te.add(Dense(num_outputs, activation=activation))
    model_te.compile(loss=loss, optimizer=optimizer, metrics = metrics_use)
    callback = tf.keras.callbacks.EarlyStopping(monitor=stop_metric, patience=patience, mode=stop_mode)
    model_te.fit(np.append(X_train ,z_target_encoded_train ,axis=1), y_train_nn,
                 validation_data=[np.append(X_val ,z_target_encoded_val ,axis=1), y_val_nn],
                 epochs=epochs, batch_size=batch_size, callbacks=[callback])
    end = time.time()

    y_train_pred_te = model_te.predict(np.append(X_train ,z_target_encoded_train ,axis=1) ,batch_size=batch_size)
    y_test_pred_te = model_te.predict(np.append(X_test ,z_target_encoded_test ,axis=1) ,batch_size=batch_size)

    eval_res_train = get_metrics(y_train_nn, y_train_pred_te, target=target)
    for metric in eval_res_train.keys():
        results["NN_te"][metric + " Train"] = eval_res_train[metric]
    eval_res_test = get_metrics(y_test_nn, y_test_pred_te, target=target)
    for metric in eval_res_test.keys():
        results["NN_te"][metric + " Test"] = eval_res_test[metric]
    results["NN_te"]["Time"] = round(end - start, 2)

    del model_te

    print(f"Train NN with OHE")
    results["NN_ohe"] = {}
    start = time.time()
    try:
        # model_ohe = tf.keras.models.clone_model(base_model)
        model_ohe, optimizer = get_model(model_name=model_name, input_size=d + z_ohe_encoded_train.shape[1],
                                        output_size=num_outputs, RS=RS)
        ### Lukas: change activation function in output layer
        model_ohe.build((n,d+z_ohe_encoded_train.shape[1]))
        update_layer_activation(model=model_ohe, activation=activation_layer)
        # model_ohe.add(Dense(num_outputs, activation=activation))
        model_ohe.compile(loss=loss, optimizer=optimizer, metrics = metrics_use)
        callback = tf.keras.callbacks.EarlyStopping(monitor=stop_metric, patience=patience, mode=stop_mode)
        model_ohe.fit(np.append(X_train ,z_ohe_encoded_train ,axis=1), y_train_nn,
                      validation_data=[np.append(X_val ,z_ohe_encoded_val ,axis=1), y_val_nn],
                      epochs=epochs, batch_size=batch_size, callbacks=[callback])
        end = time.time()

        y_train_pred_ohe = model_ohe.predict(np.append(X_train ,z_ohe_encoded_train ,axis=1) ,batch_size=batch_size)
        y_test_pred_ohe = model_ohe.predict(np.append(X_test ,z_ohe_encoded_test ,axis=1) ,batch_size=batch_size)

        eval_res_train = get_metrics(y_train_nn, y_train_pred_ohe, target=target)
        for metric in eval_res_train.keys():
            results["NN_ohe"][metric + " Train"] = eval_res_train[metric]
        eval_res_test = get_metrics(y_test_nn, y_test_pred_ohe, target=target)
        for metric in eval_res_test.keys():
            results["NN_ohe"][metric + " Test"] = eval_res_test[metric]
        results["NN_ohe"]["Time"] = round(end - start, 2)

        del model_ohe
    except:
        print("Training of OHE model failed - probably there were too many categories")
        eval_res_train = get_metrics(y_train, np.zeros(y_train_nn.shape[0]), target=target)
        for metric in eval_res_train.keys():
            results["NN_ohe"][metric + " Train"] = eval_res_train[metric]
        eval_res_test = get_metrics(y_test, np.zeros(y_test_nn.shape[0]), target=target)
        for metric in eval_res_test.keys():
            results["NN_ohe"][metric + " Test"] = eval_res_test[metric]
        results["NN_ohe"]["Time"] = 0

    print(f"Train NN with Embeddings")
    results["NN_embed"] = {}
    start = time.time()
    if config["embed_parameters"]["embed_dims_method"]=="sqrt":
        embed_dims = [int(np.sqrt(q)) for q in qs]
    elif config["embed_parameters"]["embed_dims_method"]=="AutoGluon":
        embed_dims = [int(np.max([100, np.round(1.6*q**0.56)])) for q in qs]
    else:
        embed_dims = [10 for q in qs]

    input_layer = Input(shape=(d,))

    # Define embedding layers
    embed_inputs = []
    embedding_layers = []
    for q_num in range(len(qs)):
        Z_input_layer = Input(shape=(1,))
        embedding_layer = Embedding(qs[q_num], embed_dims[q_num], input_length=1)(Z_input_layer)
        embedding_layer = Reshape(target_shape=(embed_dims[q_num],))(embedding_layer)

        embed_inputs.append(Z_input_layer)
        embedding_layers.append(embedding_layer)

    if model_name=="AutoGluon":
        ### Get model layer dimensions
        min_numeric_embed_dim = 32
        max_numeric_embed_dim = 2056
        max_layer_width = 2056
        # Main dense model
        if target == "continuous":
            default_layer_sizes = [256,
                                   128]  # overall network will have 4 layers. Input layer, 256-unit hidden layer, 128-unit hidden layer, output layer.
        else:
            default_sizes = [256, 128]  # will be scaled adaptively
            # base_size = max(1, min(num_net_outputs, 20)/2.0) # scale layer width based on number of classes
            base_size = max(1, min(num_outputs,
                                   100) / 50)  # TODO: Updated because it improved model quality and made training far faster
            default_layer_sizes = [defaultsize * base_size for defaultsize in default_sizes]
        layer_expansion_factor = 1  # TODO: consider scaling based on num_rows, eg: layer_expansion_factor = 2-np.exp(-max(0,train_dataset.num_examples-10000))
        first_layer_width = int(min(max_layer_width, layer_expansion_factor * default_layer_sizes[0]))

        # numeric embed dim
        vector_dim = 0  # total dimensionality of vector features (I think those should be transformed string features, which we don't have)
        prop_vector_features = perc_numeric  # Fraction of features that are numeric
        numeric_embedding_size = int(min(max_numeric_embed_dim,
                                         max(min_numeric_embed_dim,
                                             first_layer_width * prop_vector_features * np.log10(vector_dim + 10))))


        numeric_embedding = Dense(numeric_embedding_size, activation="relu")(input_layer)

        concat = Concatenate()([numeric_embedding] + embedding_layers)

        base_model, optimizer = get_model(model_name="AutoGluon_no_numeric", input_size=numeric_embedding_size + sum(embed_dims), output_size=num_outputs, target=target,
                                          perc_numeric=perc_numeric, RS=RS)
        base_model.build((n, numeric_embedding_size + sum(embed_dims)))
        update_layer_activation(model=base_model, activation=activation_layer)

    else:
        # load base model and change output layer activation
        # baseline_model = tf.keras.models.clone_model(base_model)
        base_model, optimizer = get_model(model_name=model_name, input_size=d + sum(embed_dims),
                                          output_size=num_outputs, RS=RS)
        base_model.build((n, d + sum(embed_dims)))
        update_layer_activation(model=base_model, activation=activation_layer)

        concat = Concatenate()([input_layer] + embedding_layers)

    layers = base_model(concat)

    model_embed = Model(inputs=[input_layer] + embed_inputs, outputs=layers)

    model_embed.compile(loss=loss, optimizer=optimizer, metrics = metrics_use)
    callback = tf.keras.callbacks.EarlyStopping(monitor=stop_metric, patience=patience, mode=stop_mode)
    model_embed.fit([X_train] + [Z_train[: ,q_num] for q_num in range(len(qs))], y_train_nn,
                    validation_data=[[X_val] + [Z_val[: ,q_num] for q_num in range(len(qs))], y_val_nn],
                    epochs=epochs, batch_size=batch_size, callbacks=[callback])
    end = time.time()

    y_train_pred_embed = model_embed.predict([X_train] + [Z_train[: ,q_num] for q_num in range(len(qs))]
                                             ,batch_size=batch_size)
    y_test_pred_embed = model_embed.predict([X_test] + [Z_test[: ,q_num] for q_num in range(len(qs))]
                                            ,batch_size=batch_size)

    eval_res_train = get_metrics(y_train_nn, y_train_pred_embed, target=target)
    for metric in eval_res_train.keys():
        results["NN_embed"][metric + " Train"] = eval_res_train[metric]
    eval_res_test = get_metrics(y_test_nn, y_test_pred_embed, target=target)
    for metric in eval_res_test.keys():
        results["NN_embed"][metric + " Test"] = eval_res_test[metric]
    results["NN_embed"]["Time"] = round(end - start, 2)

    del model_embed

    if target=="continuous" or (target=="binary" and len(qs)==1):
        print(f"Train LMMNN")
        results["LMMNN"] = {}
        results["LMMNN_FE"] = {}
        if target=="binary":
            mode = 'glmm'
        elif target == "continuous":
            mode = 'intercepts'

        X_train_lmmnn = pd.concat([X_train,pd.DataFrame(Z_train,index=X_train.index,columns=[f"z{q_num}" for q_num in range(Z_train.shape[1])])],axis=1)
        X_val_lmmnn = pd.concat([X_val,pd.DataFrame(Z_val,index=X_val.index,columns=[f"z{q_num}" for q_num in range(Z_val.shape[1])])],axis=1)
        X_test_lmmnn =pd.concat([X_test,pd.DataFrame(Z_test,index=X_test.index,columns=[f"z{q_num}" for q_num in range(Z_test.shape[1])])],axis=1)
        if target == "binary":
            y_train_lmmnn = pd.Series(y_train.astype(int),index=X_train_lmmnn.index)
            y_val_lmmnn = pd.Series(y_val.astype(int),index=X_val_lmmnn.index)
            y_test_lmmnn = pd.Series(y_test.astype(int),index=X_test_lmmnn.index)
        else:
            y_train_lmmnn = pd.Series(y_train.astype(np.float32),index=X_train_lmmnn.index)
            y_val_lmmnn = pd.Series(y_val.astype(np.float32),index=X_val_lmmnn.index)
            y_test_lmmnn = pd.Series(y_test.astype(np.float32),index=X_test_lmmnn.index)


        qs_lmmnn = [i.max()+1 for i in np.concatenate([Z_train,Z_val,Z_test]).transpose()]

        start = time.time()
        model_lmmnn_fe, optimizer = get_model(model_name=model_name, input_size=d,
                                              output_size=num_outputs, RS=RS)

        x_cols = list(X_train.columns)
        q_spatial =[]
        n_neurons = [100, 50]
        dropout = None
        activation = 'relu'
        n_sig2bs = len(qs_lmmnn)
        n_sig2bs_spatial = 0
        est_cors = []
        dist_matrix = None
        time2measure_dict = None
        spatial_embed_neurons = None
        verbose = True
        Z_non_linear = False
        log_params = False
        idx = None
        Z_embed_dim_pct = 10

        y_pred_test_lmmnn, sigmas, rhos, weibull, n_epochs, b_hat, model_lmmnn, y_pred_train_lmmnn = reg_nn_lmm(
            model_lmmnn_fe, optimizer,
            X_train_lmmnn, X_val_lmmnn, X_test_lmmnn, y_train_lmmnn, y_val_lmmnn, y_test_lmmnn, qs_lmmnn, q_spatial, x_cols,
            batch_size, epochs, patience,
            n_neurons, dropout, activation, mode,
            n_sig2bs, n_sig2bs_spatial, est_cors, dist_matrix, spatial_embed_neurons, verbose, Z_non_linear,
            Z_embed_dim_pct, log_params, idx)
        end = time.time()

        y_train_pred_lmmnn_fe = model_lmmnn.predict \
            ([X_train, np.zeros(y_train_lmmnn.shape[0]), [Z_train[: ,num] for num in range(len(qs_lmmnn))]], batch_size = X_train.shape[0]).ravel()
        y_test_pred_lmmnn_fe = model_lmmnn.predict \
            ([X_test, np.zeros(y_test_lmmnn.shape[0]), [Z_test[: ,num] for num in range(len(qs_lmmnn))]], batch_size = X_test.shape[0]).ravel()

        b_hats = [b_hat[:qs_lmmnn[num]] if num == 0 else b_hat[sum(qs_lmmnn[:num]):] if num==len(qs_lmmnn) -1 else b_hat[sum(qs_lmmnn[:num]):sum
            (qs_lmmnn[:num] ) +qs_lmmnn[num]] for num in range(len(qs_lmmnn))]

        y_train_pred_lmmnn = y_train_pred_lmmnn_fe + tf.reduce_sum([tf.gather(b_hat, Z_train[: ,num]) for num, b_hat in enumerate(b_hats)], axis=0).numpy()
        y_test_pred_lmmnn = y_test_pred_lmmnn_fe + tf.reduce_sum([tf.gather(b_hat, Z_test[: ,num]) for num, b_hat in enumerate(b_hats)], axis=0).numpy()
        if target == "binary":
            y_train_pred_lmmnn = sigmoid(y_train_pred_lmmnn)
            y_test_pred_lmmnn = sigmoid(y_test_pred_lmmnn)
            y_train_pred_lmmnn_fe = sigmoid(y_train_pred_lmmnn_fe)
            y_test_pred_lmmnn_fe = sigmoid(y_test_pred_lmmnn_fe)

        eval_res_train = get_metrics(y_train, y_train_pred_lmmnn, target=target)
        for metric in eval_res_train.keys():
            results["LMMNN"][metric + " Train"] = eval_res_train[metric]
        eval_res_test = get_metrics(y_test, y_test_pred_lmmnn, target=target)
        for metric in eval_res_test.keys():
            results["LMMNN"][metric + " Test"] = eval_res_test[metric]
        results["LMMNN"]["Time"] = round(end - start, 2)

        eval_res_train_fe = get_metrics(y_train, y_train_pred_lmmnn_fe, target=target)
        for metric in eval_res_train_fe.keys():
            results["LMMNN_FE"][metric + " Train"] = eval_res_train_fe[metric]
        eval_res_test_fe = get_metrics(y_test, y_test_pred_lmmnn_fe, target=target)
        for metric in eval_res_test_fe.keys():
            results["LMMNN_FE"][metric + " Test"] = eval_res_test_fe[metric]
        results["LMMNN_FE"]["Time"] = round(end - start, 2)

        model_lmmnn_info = {"sigmas": sigmas,
                            "b_hats": b_hats}

        del model_lmmnn, model_lmmnn_fe

    results_dict= {"results": results}

    if target=="continuous" or (target=="binary" and len(qs)==1):
        results_dict["model_lmmnn_info"] = model_lmmnn_info

    if save_results:
        with open(f"{save_path}/results_dict.pickle", 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results_dict

def train_lowcard_models(train_data, val_data, test_data, config, RS=42, save_results=True, save_path=""):
    tf.random.set_seed(RS)
    # Prepare results
    results = {}

    # Check whether all directories are created
    if save_results:
        save_path_split = save_path.split("/")
        for i in range(len(save_path_split)):
            if not os.path.exists("/".join(save_path_split[:i + 1])):
                os.mkdir("/".join(save_path_split[:i + 1]))


    # Get data
    X_train, Z_train, y_train, z_ohe_encoded_train, z_target_encoded_train = train_data
    X_val, Z_val, y_val, z_ohe_encoded_val, z_target_encoded_val = val_data
    X_test, Z_test, y_test, z_ohe_encoded_test, z_target_encoded_test = test_data

    # Load parameters from config
    # General parameters
    target = config["general_parameters"]["target"]
    if target=="binary":
        activation = "sigmoid"
        activation_layer = tf.keras.activations.sigmoid
        loss = tf.keras.losses.BinaryCrossentropy()
        xgb = XGBClassifier
        lr = LogisticRegression(solver='lbfgs', max_iter=10000)
        num_outputs = 1
    elif target=="continuous":
        activation = "linear"
        activation_layer = tf.keras.activations.linear
        loss = tf.keras.losses.MeanSquaredError()
        xgb = XGBRegressor
        lr = LinearRegression()
        num_outputs = 1
    elif target == "categorical":
        activation = "softmax"
        activation_layer = tf.keras.activations.softmax
        loss = tf.keras.losses.CategoricalCrossentropy()
        xgb = XGBClassifier
        lr = LogisticRegression(solver='lbfgs', max_iter=10000)
        num_outputs = np.unique(y_train).shape[0]

    if target == "categorical":
        y_train_nn = tf.one_hot(y_train,num_outputs)
        y_val_nn = tf.one_hot(y_val,num_outputs)
        y_test_nn = tf.one_hot(y_test,num_outputs)
    elif target in ["continuous", "binary"]:
        y_train_nn = y_train
        y_val_nn = y_val
        y_test_nn = y_test


    metrics = config["general_parameters"]["metrics"]
    model_name = config["general_parameters"]["model_name"]


    # NN parameters
    epochs = config["nn_parameters"]["epochs"]
    batch_size = config["nn_parameters"]["batch_size"]
    patience = config["nn_parameters"]["patience"]
    stop_metric = config["nn_parameters"]["stop_metric"]

    # AutoGluon specific parameters

    # Infer necessary parameters
    qs = [np.unique(i).shape[0] for i in Z_train.transpose()]
    d = X_train.shape[1] # columns
    n = X_train.shape[0] # rows
    perc_numeric = d/(d+Z_train.shape[1])

    # Define base model
    print(f"Load base model")
    ### Lukas: changed get_model call with new parameter output_size
    base_model, optimizer = get_model(model_name=model_name, input_size=d, output_size=num_outputs, target=target, perc_numeric=perc_numeric, RS=RS)

    print(f"Train XGBoost without z features")
    start = time.time()
    if target=="categorical":
        xgb_metric = "merror"
    if target in ["binary", "continuous"]:
        xgb_metric = "error"

    if X_train.shape[1 ]!=0:
        results["XGB"] = {}
        xgb_model = xgb(eval_metric=xgb_metric)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val ,y_val)], verbose=False)
        end = time.time()
        if target == "binary":
            y_train_pred_xgb = xgb_model.predict_proba(X_train)[:,1]
            y_test_pred_xgb = xgb_model.predict_proba(X_test)[:,1]
        elif target == "categorical":
            y_train_pred_xgb = xgb_model.predict_proba(X_train)
            y_test_pred_xgb = xgb_model.predict_proba(X_test)
        elif target=="continuous":
            y_train_pred_xgb = xgb_model.predict(X_train)
            y_test_pred_xgb = xgb_model.predict(X_test)

        eval_res_train = get_metrics(y_train_nn, y_train_pred_xgb, target=target)
        for metric in eval_res_train.keys():
            results["XGB"][metric + " Train"] = eval_res_train[metric]
        eval_res_test = get_metrics(y_test_nn, y_test_pred_xgb, target=target)
        for metric in eval_res_test.keys():
            results["XGB"][metric + " Test"] = eval_res_test[metric]
        results["XGB"]["Time"] = round(end - start, 2)

    else:
        eval_res_train = get_metrics(y_train, np.zeros(y_train.shape[0]), target=target)
        for metric in eval_res_train.keys():
            results["XGB"][metric + " Train"] = eval_res_train[metric]
        eval_res_test = get_metrics(y_test, np.zeros(y_test.shape[0]), target=target)
        for metric in eval_res_test.keys():
            results["XGB"][metric + " Test"] = eval_res_test[metric]
        results["XGB"]["Time"] = 0

    print(f"Train Linear Model without z features")
    start = time.time()
    if X_train.shape[1 ]!=0:
        results["LR"] = {}
        lr_model = lr
        lr_model.fit(X_train, y_train)
        end = time.time()
        if target == "binary":
            y_train_pred_lr = lr_model.predict_proba(X_train)[:,1]
            y_test_pred_lr = lr_model.predict_proba(X_test)[:,1]
        elif target == "categorical":
            y_train_pred_lr = lr_model.predict_proba(X_train)
            y_test_pred_lr = lr_model.predict_proba(X_test)
        elif target=="continuous":
            y_train_pred_lr = lr_model.predict(X_train)
            y_test_pred_lr = lr_model.predict(X_test)

        eval_res_train = get_metrics(y_train_nn, y_train_pred_lr, target=target)
        for metric in eval_res_train.keys():
            results["LR"][metric + " Train"] = eval_res_train[metric]
        eval_res_test = get_metrics(y_test_nn, y_test_pred_lr, target=target)
        for metric in eval_res_test.keys():
            results["LR"][metric + " Test"] = eval_res_test[metric]
        results["LR"]["Time"] = round(end - start, 2)

    else:
        eval_res_train = get_metrics(y_train, np.zeros(y_train.shape[0]), target=target)
        for metric in eval_res_train.keys():
            results["LR"][metric + " Train"] = eval_res_train[metric]
        eval_res_test = get_metrics(y_test, np.zeros(y_test.shape[0]), target=target)
        for metric in eval_res_test.keys():
            results["LR"][metric + " Test"] = eval_res_test[metric]
        results["LR"]["Time"] = 0


    metrics_use = []
    if "auc" in metrics:
        metrics_use.append(tf.keras.metrics.AUC(from_logits=True, name="auc"))
    if "accuracy" in metrics:
        if target =="binary":
            metrics_use.append(tf.keras.metrics.Accuracy(name="accuracy"))
        elif target == "categorical":
            metrics_use.append(tf.keras.metrics.CategoricalAccuracy(name="accuracy"))
    if "f1" in metrics:
        if target == "binary":
            metrics_use.append(F1Score(num_classes=2, average="micro", name="f1"))
        elif target == "categorical":
            metrics_use.append(F1Score(num_classes=num_outputs, average="weighted", name="f1"))
    if "r2" in metrics:
        metrics_use.append(RSquare(name="r2"))
    if "mse" in metrics:
        metrics_use.append(tf.keras.metrics.MeanSquaredError(name="mse"))
    if stop_metric in ["auc", "accuracy", "f1", "r2", "val_auc", "val_accuracy", "val_f1", "val_r2"]:
        stop_mode = "max"
    else:
        stop_mode = "min"

    print(f"Train NN without Z features")
    results["NN"] = {}
    start = time.time()
    if X_train.shape[1 ]!=0:
        model_nn = tf.keras.models.clone_model(base_model)
        ### Lukas: change activation function in output layer
        model_nn.build((n,d))
        update_layer_activation(model=model_nn, activation=activation_layer)
        # model_nn.add(Dense(num_outputs, activation=activation))
        model_nn.compile(loss=loss, optimizer=optimizer, metrics = metrics_use)
        callback = tf.keras.callbacks.EarlyStopping(monitor=stop_metric, patience=patience, mode=stop_mode)
        model_nn.fit(X_train, y_train_nn,
                     validation_data= [X_val, y_val_nn],
                     epochs=epochs, batch_size=batch_size, callbacks=[callback])
        end = time.time()

        y_train_pred_nn = model_nn.predict(X_train ,batch_size=batch_size)
        y_test_pred_nn = model_nn.predict(X_test ,batch_size=batch_size)

        eval_res_train = get_metrics(y_train_nn, y_train_pred_nn, target=target)
        for metric in eval_res_train.keys():
            results["NN"][metric + " Train"] = eval_res_train[metric]
        eval_res_test = get_metrics(y_test_nn, y_test_pred_nn, target=target)
        for metric in eval_res_test.keys():
            results["NN"][metric + " Test"] = eval_res_test[metric]
        results["NN"]["Time"] = round(end - start, 2)

        del model_nn
    else:
        eval_res_train = get_metrics(y_train_nn, np.zeros(y_train_nn.shape[0]), target=target)
        for metric in eval_res_train.keys():
            results["NN"][metric + " Train"] = eval_res_train[metric]
        eval_res_test = get_metrics(y_test_nn, np.zeros(y_test_nn.shape[0]), target=target)
        for metric in eval_res_test.keys():
            results["NN"][metric + " Test"] = eval_res_test[metric]
        results["NN"]["Time"] = 0



    results_dict= {"results": results}

    if save_results:
        with open(f"{save_path}/results_dict.pickle", 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results_dict









