# -*- coding: utf-8 -*-
"""
Train word recognition model from wordDataset.
Same pipeline as alphabet model: right hand only, 42 features, scale+center.

Run from backend/ directory after preparing wordDataset:
    python -m PSL.word_recognition.train_word_model
"""

import sys
import sqlite3
import pickle
import numpy as np

from sklearn import preprocessing
from tracking.data_splits import split_three_way
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback


from tracking.training import BACKEND, TrainingRun
from tracking.data_validation import validate_training_database

DATASET_PATH = BACKEND / "data" / "db" / "main_dataset.db"


def train_words():
    validation = validate_training_database(DATASET_PATH, "wordDataset")
    with TrainingRun("word", __file__) as run:
        run.metadata["data_validation"] = validation
        return _train_words(run)


def _train_words(run):
    connection = sqlite3.connect(DATASET_PATH.as_uri() + "?mode=ro", uri=True)
    crsr = connection.cursor()

    cols = ",".join([f"x{i},y{i}" for i in range(1, 22)])
    crsr.execute(f"SELECT {cols}, label FROM wordDataset")
    rows = crsr.fetchall()
    connection.close()

    if not rows:
        print("No data found in wordDataset. Run images_to_word_db.py first.")
        sys.exit(1)

    features = [list(r[:-1]) for r in rows]
    labels   = [r[-1] for r in rows]

    le = preprocessing.LabelEncoder()
    label_encoded = le.fit_transform(labels)
    num_classes = len(set(labels))
    print(f"Classes ({num_classes}): {list(le.classes_)}")
    print(f"Total samples: {len(features)}")

    label_cat = to_categorical(label_encoded, num_classes=num_classes)

    train_idx, val_idx, test_idx = split_three_way(features, labels)
    X = np.asarray(features, dtype=np.float64)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = label_cat[train_idx], label_cat[val_idx], label_cat[test_idx]

    run.record_dataset(features, labels, DATASET_PATH, "wordDataset",
                       train_samples=len(X_train), test_samples=len(X_test))

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    with (run.output_dir / "word_scaler.pkl").open("wb") as stream:
        pickle.dump(scaler, stream)
    with (run.output_dir / "word_label_encoder.pkl").open("wb") as stream:
        pickle.dump(le, stream)

    model = Sequential([
        Dense(256, activation='relu', input_shape=(42,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64,  activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    run.record_model(model, epochs=50, batch_size=32, split={
        "test_size": 0.2, "validation_size": 0.2,
        "split_random_state": 42, "duplicate_disjoint": True,
    })

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=32, verbose=1,
              callbacks=[run.callback(Callback)])

    score = model.evaluate(X_test, y_test, verbose=0)
    run.record_test(loss=score[0], accuracy=score[1])
    print(f"\nTest accuracy: {score[1]*100:.2f}%")

    model.save(run.output_dir / "word_model.h5")
    print(f"Saved: {run.output_dir / 'word_model.h5'}")


if __name__ == "__main__":
    train_words()
