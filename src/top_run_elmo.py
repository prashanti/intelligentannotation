from networkx.algorithms.traversal.depth_first_search import dfs_tree

import tensorflow.compat.v1 as tf  # type: ignore
from tensorflow.compat.v1.keras.backend import set_session  # type: ignore

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import argparse
import sys
import json
import logging
import os
import numpy as np
import pandas as pd

import networkx as nx
import tensorflow_addons as tfa
import tensorflow_hub as hub

from tensorflow.keras.preprocessing.sequence import (  # type: ignore
    pad_sequences)

from keras.models import Model
from keras.layers import Lambda
from keras.layers import Input, Embedding, Dense, TimeDistributed, Dropout,\
    Bidirectional, concatenate, SpatialDropout1D, GRU
from tensorflow.keras.utils import to_categorical  # type: ignore


tf.disable_v2_behavior()


parser = argparse.ArgumentParser(
    description="Parameters and Hyperparameters as input",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-w", "--weight", type=float, default=0.5,
    help="Weight between 0 and 1 for semantic similarity"
    " output embedding vector")
parser.add_argument(
    "-lr", "--learning_rate", type=float, default=0.001,
    help="Learning rate for model optimizer")
parser.add_argument(
    "-e", "--epochs", type=int, default=35,
    help="Epochs for model training")
parser.add_argument(
    "-b", "--batch_size", type=int, default=64,
    help="Batch size for each epoch")
parser.add_argument(
    "-a", "--activation", type=str, default='softmax',
    choices=['softmax', 'sigmoid', 'relu'],
    help="Activation for output layer")
parser.add_argument(
    "-r", "--rdropout", type=float, default=0.3,
    help="Recurrent dropout for bidirectional GRU")
parser.add_argument(
    "-o", "--optimizer", type=str, default='adamw',
    choices=['adam', 'adamw', 'rmsprop'],
    help="Optimizer for loss optimization")
parser.add_argument(
    "-l", "--loss", type=str, default='sigfocalCE',
    choices=[
        'categoricalCE', 'sigfocalCE', 'sparsecatCE',
        'kl', 'binaryCE', 'mse'],
    help="Loss function")
parser.add_argument(
    "-ml", "--max_len", type=int, default=71,
    help="Maximum number of tokens to keep for each "
    "sentence in the corpus")
parser.add_argument(
    "-cl", "--max_char_len", type=int, default=15,
    help="Maximum number of characters to keep from each "
    "token")
parser.add_argument(
    "-sl", "--min_sent_len", type=int, default=3,
    help="Keep a sentence for training if number of tokens "
    "> min_sent_len")
parser.add_argument(
    "-d", "--dropout", type=float, default=0.5,
    help="Dropout rate in the hidden layers")
parser.add_argument(
    "-n", "--new_log", type=bool, default=False,
    help="Create a new log or append on existing log")

args = parser.parse_args()
config = vars(args)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "model_output", "Experiments")
DATASET_LOC = os.path.join(DATA_DIR, "model_input", "dataset")
direct_parent = os.path.join(DATA_DIR, "GO_Category", "GO_DirectParents.tsv")
go_category = ["GO:0008150", "GO:0005575", "GO:0003674"]

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if config.get('new_log'):
    with open(os.path.join(LOG_DIR, "elmo.log"), 'w+') as f:
        pass

file_handler = logging.FileHandler(filename=os.path.join(LOG_DIR, "elmo.log"))
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(
    handlers=handlers,
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p')


def to_log(str, num=23):
    logger.info(str.replace("\n", "\n{0}".format(" "*num)))


to_log("Download ELMO from tensorflow hub")
elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)


def get_sim(term1, term2):
    if "GO" in term1 and "GO" in term2:
        term1 = term1.replace("B-", "").replace("I-", "")
        term2 = term2.replace("B-", "").replace("I-", "")
        t1 = set(subsumers.get(term1, term1))
        t2 = set(subsumers.get(term2, term2))
        if len(set.union(t1, t2)) > 0:
            simj = len(set.intersection(t1, t2)) / len(set.union(t1, t2))
        else:
            simj = 0.0
    else:
        simj = 0.0
    return simj


def get_optimizer(opt, lr):
    if opt == 'adam':
        return tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=0.9, beta_2=0.999)
    elif opt == 'adamw':
        step = tf.Variable(0, trainable=False)
        schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [10000, 15000], [1e-0, 1e-1, 1e-2])
        # lr and wd can be a function or a tensor
        lr = lr * schedule(step)

        def wd():
            return 1e-4 * schedule(step)
        # wd = lambda: 1e-4 * schedule(step)  # type: ignore
        return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    elif opt == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=lr)


def get_loss(loss):
    if loss == 'categoricalCE':
        return tf.keras.losses.CategoricalCrossentropy()
    elif loss == 'sigfocalCE':
        return tfa.losses.SigmoidFocalCrossEntropy()
    elif loss == 'sparsecatCE':
        return tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False),
    elif loss == 'kl':
        return tf.keras.losses.KLDivergence()
    elif loss == 'binaryCE':
        return tf.keras.losses.BinaryCrossentropy()
    elif loss == 'mse':
        return tf.keras.losses.MeanSquaredError()


def ElmoEmbedding(x):
    return elmo_model(
        inputs={
            "tokens": tf.squeeze(tf.cast(x, tf.string)),
            "sequence_len": tf.cast(
                tf.count_nonzero(x, axis=1),
                dtype=tf.int32)
        },
        signature="tokens",
        as_dict=True)["elmo"]


if __name__ == '__main__':
    to_log('Creating ontology heirarchy')
    direct_data = pd.read_csv(
        direct_parent, delimiter="\t",
        names=['Child', 'Parent']).replace({"_": ":"}, regex=True)
    direct_data = direct_data.drop(0).reset_index(drop=True)
    onto_digraph = nx.from_pandas_edgelist(
        direct_data, source='Child', target='Parent',
        create_using=nx.classes.digraph.DiGraph)
    onto_info = "Number of nodes: {0}\nNumber of edges: {1}".format(
        onto_digraph.number_of_nodes(),
        onto_digraph.number_of_edges(),
    )
    to_log(onto_info)
    to_log('Creating list of subsumers')
    subsumers = dict(
        (i, list(
            set(np.array(dfs_tree(onto_digraph, i).edges()).flatten().tolist()
                + [i]) - set(["owl:Thing"])
            )) for i in onto_digraph.nodes())

    to_log('Parameters and hyperparameters for model training')
    config.update({
        "callbacks": [
            "to_log"
        ],
        "project": "Intelligent_OA",
        "extra_info": "ELMo, inputs: Word(30D), POS(100D)",
        "name": "ELMO",
    })
    to_log(json.dumps(config, indent=4))
    train_data = json.load(open(os.path.join(DATASET_LOC, "train.json"), "r"))
    train_data = [
        i for i in train_data if len(i['tokens']) >= config.get("min_sent_len")
    ]

    test_data = json.load(open(os.path.join(DATASET_LOC, "test.json"), "r"))
    test_data = [
        i for i in test_data if len(i['tokens']) >= config.get("min_sent_len")
    ]
    input_data = train_data + test_data

    all_data = {
        "tokens": [i['tokens'] for i in input_data],
        "tags": [i['iob_tags'] for i in input_data],
        "pos": [i['pos_tags'] for i in input_data],
    }
    assert (
        len(all_data['tokens'])
        == len(all_data['tags'])
        == len(all_data['pos'])
    )

    to_log('Creating training and test dataset')
    words = ["PAD"] + sorted(
        set([j for i in all_data['tokens'] for j in i] + ["UNK", "O"])
        - set(["PAD"]))
    tags = ["PAD"] + sorted(
        set([j for i in all_data['tags'] for j in i] + ["UNK", "O"])
        - set(["PAD"]))
    chars = ["PAD"] + sorted(
        set([j for i in words for j in i] + ["UNK", "O"])
        - set(["PAD"]))
    pos = ["PAD"] + sorted(
        set([j for i in all_data['pos'] for j in i] + ["UNK", "O"])
        - set(["PAD"]))

    n_words, n_tags, n_chars, = len(words), len(tags), len(chars)
    n_pos = len(pos)

    corpus_info = (
        "\nNumber of Observations:{0}\nNumber of words:{1}"
        "\nNumber of tags:{2}\nNumber of characters: {3}"
        "\nNumber of pos: {4}".format(
            len(all_data['tokens']), n_words, n_tags, n_chars, n_pos)
        )
    to_log(corpus_info)

    word_to_idx = dict((i, idx) for idx, i in enumerate(words))
    idx_to_word = dict((v, k) for k, v in word_to_idx.items())

    tag_to_idx = dict((i, idx) for idx, i in enumerate(tags))
    idx_to_tag = dict((v, k) for k, v in tag_to_idx.items())

    char_to_idx = dict((i, idx) for idx, i in enumerate(chars))
    idx_to_char = dict((v, k) for k, v in char_to_idx.items())

    pos_to_idx = dict((i, idx) for idx, i in enumerate(pos))
    idx_to_pos = dict((v, k) for k, v in pos_to_idx.items())

    to_log('Creating output labels: one hot encodings')
    Y_tags = [[tag_to_idx.get(i) for i in sent] for sent in all_data['tags']]
    Y_tags = pad_sequences(
        maxlen=config.get("max_len"), sequences=Y_tags,
        value=tag_to_idx.get("PAD"), padding='post',
        truncating='post', dtype='float16')
    Y_tags = to_categorical(Y_tags, num_classes=n_tags, dtype='float16')

    to_log('Creating semantic embedding from subsumers information')
    sem_dist = dict(
        [(i, to_categorical(i, num_classes=n_tags)) for i in range(n_tags)])
    factor = 0
    for i in range(n_tags):
        iob_i = None
        term_i = idx_to_tag.get(i)
        if "B-" in term_i or "I-" in term_i:
            iob_i = term_i[0]
        term_i = term_i.replace("B-", "").replace("I-", "")
        if "GO" in term_i:
            sem_scores = []
            for j in range(n_tags):
                iob_j = None
                term_j = idx_to_tag.get(j)
                if "B-" in term_j or "I-" in term_j:
                    iob_j = term_j[0]
                term_j = term_j.replace("B-", "").replace("I-", "")
                score = config.get('weight') * get_sim(term_i, term_j)
                if iob_i != iob_j:
                    score = factor * score
                sem_scores.append(score)
            sem_scores = np.array(sem_scores)
            sem_scores[i] = 1
            sem_dist[i] = sem_scores

    for i in range(n_tags):
        num_max = np.where(sem_dist[i] == 1)[0].size
        assert num_max == 1

    for i in range(len(Y_tags)):
        for j in range(config.get("max_len")):
            k = np.where(Y_tags[i][j] == 1)[0][0]
            Y_tags[i][j] = sem_dist[k]

    to_log('Creating input dataset')
    X_word = [(s, len(s)) for s in all_data['tokens']]
    X_word = np.array(
        [
            i[0][:config.get("max_len")]
            + [""]*(config.get("max_len")-i[1]) for i in X_word
        ])

    X_char_temp = []
    for wds in all_data['tokens']:
        wds = wds[:config.get("max_len")] + ["PAD"]*(
            config.get("max_len") - len(wds))
        chrs = [list(word)[:config.get("max_char_len")] + ["PAD"]*(
            config.get("max_char_len")-len(word))
            if word != "PAD" else ["PAD"]*config.get("max_char_len")
            for word in wds]
        X_char_temp.append(np.array(chrs))
    X_char_temp = np.array(X_char_temp)
    X_char = np.vectorize(char_to_idx.get)(X_char_temp).astype('float16')
    del X_char_temp

    X_pos = [[pos_to_idx.get(w) for w in s] for s in all_data['pos']]
    X_pos = pad_sequences(
        maxlen=config.get("max_len"), sequences=X_pos,
        value=pos_to_idx.get("PAD"), padding='post', truncating='post',
        dtype='float16')

    max_idx = 0
    for i in range(len(X_word), 0, -1):
        if (
            i * 0.7 % config.get("batch_size") == 0 and
            i * 0.3 % config.get("batch_size") == 0
        ):
            max_idx = i
            break

    combined = [(X_word[i], X_char[i], X_pos[i]) for i in range(max_idx)]
    Y_tags = Y_tags[:max_idx]

    to_log('Dividing dataset into 80-20 split')
    X_tr, X_te, y_tr, y_te = train_test_split(
        combined, Y_tags, test_size=0.2, random_state=2022)

    input_train = []
    for i in range(len(combined[0])):
        temp = []
        for j in range(len(X_tr)):
            temp.append(X_tr[j][i])
        if i == 0:
            input_train.append(np.array(temp, dtype='str'))
        else:
            input_train.append(np.array(temp, dtype='float16'))

    input_test = []
    for i in range(len(combined[0])):
        temp = []
        for j in range(len(X_te)):
            temp.append(X_te[j][i])
        if i == 0:
            input_test.append(np.array(temp, dtype='str'))
        else:
            input_test.append(np.array(temp, dtype='float16'))

    to_log('Defining deep learning architecture')

    # input and embedding for words
    word_in = Input(
        shape=(config.get("max_len"),), dtype=tf.string, name="WORD")
    emb_word = Lambda(ElmoEmbedding)(word_in)

    # input and embeddings for characters
    char_in = Input(
        shape=(config.get("max_len"), config.get("max_char_len"),),
        name="CHAR")
    emb_char = TimeDistributed(
        Embedding(
            input_dim=n_chars, output_dim=100,
            input_length=config.get("max_char_len"),
            mask_zero=True))(char_in)
    char_enc = TimeDistributed(
        GRU(
            units=150, return_sequences=False, recurrent_dropout=0.5)
            )(emb_char)

    # input and embeddings for pos
    pos_in = Input(shape=(config.get("max_len"),), name="POS")
    emb_pos = Embedding(
        input_dim=n_pos, output_dim=100, input_length=config.get("max_len"),
        mask_zero=True, name="EMB_POS")(pos_in)

    # main LSTM
    x = concatenate([emb_word, char_enc, emb_pos])
    x = SpatialDropout1D(config.get("dropout"))(x)
    main_lstm = Bidirectional(
        GRU(
            units=150, return_sequences=True,
            recurrent_dropout=config['rdropout']))(x)
    main_lstm = TimeDistributed(Dense(3200, activation='relu'))(main_lstm)
    main_lstm = Dropout(config.get("dropout"))(main_lstm)
    out = TimeDistributed(
        Dense(
            n_tags, activation=config['activation']
        ), name="OUT_TAGS")(main_lstm)

    model = Model([word_in, char_in, pos_in], out, name="ELMO")
    model.compile(
        optimizer=get_optimizer(config['optimizer'], config['learning_rate']),
        loss=get_loss(config['loss']), metrics=["acc"])
    model_str = []
    model.summary(print_fn=lambda x: model_str.append(x))
    model_str = "\n".join(model_str)
    to_log(model_str)

    sess = tf.Session()
    set_session(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    model_arch = tf.keras.utils.plot_model(
        model, to_file='elmo_architecture.png', show_shapes=True)

    to_log('Starting model training')

    callbacks = []
    if config.get('callbacks'):
        assert type(config.get('callbacks')) == list
        for i in config.get('callbacks'):
            if i == "early_stop":
                callbacks.append(
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        verbose=1,
                        restore_best_weights=True,
                    )
                )
            if i == "to_log":
                callbacks.append(
                    tf.keras.callbacks.LambdaCallback(
                        on_epoch_end=lambda epoch, logs: to_log(
                            "Epoch: {0} - loss: {1:.4f} - acc: {2:.4f}".format(
                                epoch, np.mean(logs['loss']), logs['acc'],
                            )
                        )
                    )
                )

    try:
        history = model.fit(
            input_train, y_tr,
            batch_size=config.get("batch_size"),
            epochs=config.get("epochs"),
            verbose=1,
            callbacks=callbacks,
        )
    except Exception:
        sess = tf.Session()
        set_session(sess)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        history = model.fit(
            input_train, y_tr,
            batch_size=config.get("batch_size"),
            epochs=config.get("epochs"),
            verbose=1,
            callbacks=callbacks,
        )
        to_log('Model training complete')

    del X_word, X_char, X_pos, Y_tags, combined

    to_log('Making predictions')
    pred = []
    step = 100
    pbar = tqdm(total=len(X_te), desc="Making predictions:")
    for i in range(int(len(X_te)/step)+1):
        inp = [j[i*step:(i+1)*step] for j in input_test]
        if inp[0].shape[0] != 0:
            temp = model.predict(inp)
        pred.append(temp)
        pbar.update(temp.shape[0])
    pred = np.concatenate(pred)
    pbar.close()

    to_log('Calculating F1 score and semantic similarity scores')
    word = input_test[0].flatten()
    ground_truth = np.vectorize(idx_to_tag.get)(
        np.argmax(y_te, axis=-1)).flatten()
    predictions = (np.vectorize(idx_to_tag.get)(
        np.argmax(pred, axis=-1))).flatten().tolist()
    two_predictions = (np.vectorize(idx_to_tag.get)(
        np.argsort(-1*pred, axis=-1)[:, :, :2])).reshape(
            pred.shape[0]*pred.shape[1], 2).tolist()

    pd_data = pd.DataFrame({
        "Word": word,
        "Ground_Truth": ground_truth,
        "Prediction": predictions,
        "Top_Two_Predictions": two_predictions,
    })

    pd_data.drop(pd_data[pd_data['Ground_Truth'] == "PAD"].index, inplace=True)
    pd_data.drop(
        pd_data[
            (pd_data["Ground_Truth"] == "O") &
            (pd_data["Prediction"] == "O")
        ].index, inplace=True)
    pd_data.drop(
        pd_data[
            (pd_data["Ground_Truth"] == "EOS") &
            (pd_data["Prediction"] == "EOS")
        ].index, inplace=True)

    pd_data['Comparison'] = pd_data.apply(
        lambda x: x[1] if x[1] in x[-1] else x[-1][-1], axis=1)

    to_log('Creating classification report')
    top_report = classification_report(
        pd_data['Ground_Truth'],
        pd_data['Prediction'],
        zero_division=False,
        digits=4,
    )
    # print(top_report)
    score_iob = (
        "F1 score: {0}\nSemantic Similarity Score: {1}".format(
            top_report.splitlines()[-1].split()[-2],
            np.round(
                pd_data[['Prediction', 'Ground_Truth']].apply(
                    lambda x: get_sim(x[0], x[1]), axis=1).mean(), 4)
        ))
    to_log('Scores with IOB tags:\n{0}'.format(score_iob))

    df1 = pd_data.copy().replace({"B-GO:": "GO:", "I-GO:": "GO:"}, regex=True)
    report = classification_report(
        df1['Ground_Truth'],
        df1['Prediction'],
        zero_division=False,
        digits=4,
    )
    score_top_one = (
        "F1 score: {0}\nSemantic Similarity Score: {1}".format(
            report.splitlines()[-1].split()[-2],
            np.round(
                df1[['Prediction', 'Ground_Truth']].apply(
                    lambda x: get_sim(x[0], x[1]), axis=1).mean(), 4)
        ))
    to_log('Score without IOB tags:\n{0}'.format(score_top_one))

    df2 = df1.copy()
    df2.drop(
        df2[
            (df2["Ground_Truth"] == "O") &
            (df2["Comparison"] == "O")
        ].index, inplace=True)
    df2.drop(
        df2[
            (df2["Ground_Truth"] == "EOS") &
            (df2["Comparison"] == "EOS")
        ].index, inplace=True)

    top_two_report = classification_report(
        df2['Ground_Truth'],
        df2['Comparison'],
        zero_division=False,
        digits=4,
    )

    score_top_two = (
        "F1 Score: {0}\nSemantic Similarity Score: {1}".format(
            top_two_report.splitlines()[-1].split()[-2],
            np.round(
                df2[['Comparison', 'Ground_Truth']].apply(
                    lambda x: get_sim(x[0], x[1]), axis=1).mean(), 4)
        ))
    to_log('Score without IOB tags for top 2 predictions:\n{0}'.format(
        score_top_two
    ))
