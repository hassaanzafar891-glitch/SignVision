import sqlite3
import numpy as np
import pickle

from sklearn import preprocessing
from tracking.data_splits import split_three_way
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import Callback


from tracking.training import BACKEND, TrainingRun

DATASET_PATH = BACKEND / "data" / "db" / "main_dataset.db"


class EpochBar(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        print(f"\nTraining for {self.epochs} epochs:\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filled = int((epoch + 1) / self.epochs * 20)
        bar = '█' * filled + '░' * (20 - filled)
        pct = (epoch + 1) / self.epochs * 100
        loss = logs.get('loss', 0)
        acc  = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc  = logs.get('val_accuracy', 0)
        print(f"Epoch {epoch+1:3d}/{self.epochs} [{bar}] {pct:5.1f}%"
              f"  loss: {loss:.4f}  acc: {acc:.4f}"
              f"  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")


def train_alphabets():
    with TrainingRun("alphabet", __file__) as run:
        return _train_alphabets(run)


def _train_alphabets(run):
    connection = sqlite3.connect(DATASET_PATH.as_uri() + "?mode=ro", uri=True)
    crsr = connection.cursor()

    sql = 'SELECT x1,y1'
    for x in range(2,22):
        sql = sql + ',x'+str(x)+',y'+str(x)
    sql = sql + ' FROM alphabetDataset WHERE 1'
    crsr.execute(sql)
    feature_res = crsr.fetchall()
    feature_res = np.asarray(feature_res)
    features=[]
    for x in feature_res:
        features.append(x)

    crsr.execute('SELECT label FROM alphabetDataset WHERE 1')
    label_res = crsr.fetchall()
    labels=[]
    for x in label_res:
        labels.append(x)

    connection.close()

    le = preprocessing.LabelEncoder()
    label_encoded=le.fit_transform(labels)

    num_classes = len(set(le.classes_))
    label_encoded = to_categorical(label_encoded, num_classes=num_classes)

    train_idx, val_idx, test_idx = split_three_way(features, np.asarray(labels).reshape(-1))
    X = np.asarray(features, dtype=np.float64)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = label_encoded[train_idx], label_encoded[val_idx], label_encoded[test_idx]

    run.record_dataset(features, labels, DATASET_PATH, "alphabetDataset",
                       train_samples=len(X_train), test_samples=len(X_test))

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    with (run.output_dir / "alphabet_scaler.pkl").open("wb") as stream:
        pickle.dump(scaler, stream)
    with (run.output_dir / "alphabet_label_encoder.pkl").open("wb") as stream:
        pickle.dump(le, stream)

    model = Sequential()

    model.add(Dense(120, activation='relu', input_shape=(42,)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    run.record_model(model, epochs=25, batch_size=1, split={
        "test_size": 0.2, "validation_size": 0.2,
        "split_random_state": 42, "duplicate_disjoint": True,
    })

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=25, batch_size=1, verbose=0,
                        callbacks=[EpochBar(), run.callback(Callback)])

    model.save(run.output_dir / "alphabet_model.h5")

    y_pred = model.predict(X_test)
    score = model.evaluate(X_test, y_test,verbose=1)
    run.record_test(loss=score[0], accuracy=score[1])

    print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy vs Epoch')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    print(model.summary())
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    import itertools
    plt.rcParams.update({'font.size': 10})

    def plot_confusion_matrix(cm, classes, normalize=False,
                              title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=14)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90, fontsize=9)
        plt.yticks(tick_marks, classes, fontsize=9)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            val = format(cm[i, j], fmt)
            if normalize and cm[i, j] < 0.01:
                continue
            plt.text(j, i, val, horizontalalignment="center", fontsize=7,
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label', fontsize=12)
        plt.xlabel('Predicted label', fontsize=12)

    l = np.array(labels)
    l = np.unique(l)
    class_names = l

    plt.figure(figsize=(40, 40))
    plot_confusion_matrix(cm, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(run.output_dir / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to {run.output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    train_alphabets()
