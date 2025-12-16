from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Network:
    def __init__(self):
        self.model = self._initialize_model()
        return

    def _initialize_model(self):
        reg = regularizers.l1_l2(l1=1e-5, l2=1e-4)

        model = Sequential(
            [
                Dense(21, activation="relu", kernel_regularizer=reg, input_shape=(21,)),
                Dropout(0.2),
                Dense(32, activation="relu", kernel_regularizer=reg),
                Dropout(0.2),
                Dense(16, activation="relu", kernel_regularizer=reg),
                Dropout(0.2),
                Dense(2, activation="softmax"),
            ]
        )

        adam = keras.optimizers.Adam(learning_rate=0.005)
        loss = keras.losses.CategoricalCrossentropy(from_logits=False)

        model.compile(
            optimizer=adam,
            loss=loss,
            metrics=[
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                keras.metrics.AUC(name="auc", multi_label=True, from_logits=False),
            ],
        )
        return model

    def _get_class_weights(self, Y_train):
        y_idx = pd.Series(Y_train.values.argmax(axis=1))
        class_counts = np.bincount(y_idx)  # Count samples per class
        total_samples = len(Y_train)
        class_weights = {
            i: total_samples / (len(class_counts) * count)  # Inverse frequency scaling
            for i, count in enumerate(class_counts)
        }
        print("Class weight:", class_weights)
        return class_weights

    def train_model(self, X_train, Y_train, batch_size=8, epochs=50, show_plot=False):
        print("X_train.shape =", X_train.shape)

        class_weights = self._get_class_weights(Y_train)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=6, verbose=0, min_lr=0.0000001
            ),
            ModelCheckpoint("checkpoints/best_model.keras", save_best_only=True),
        ]

        history = self.model.fit(
            X_train,
            Y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=0,
        )
        if show_plot:
            # summarize history for accuracy
            plt.plot(history.history["accuracy"])
            plt.plot(history.history["val_accuracy"])
            plt.title("model accuracy")
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.legend(["train", "val"], loc="upper left")
            plt.show()

        return self.model

    def evaluate_model(self, X_test, Y_test, print_scores=True):
        test_scores = self.model.evaluate(X_test, Y_test, verbose=0)
        if print_scores:
            print(
                "Test Loss:",
                test_scores[0],
                "Test Accuracy:",
                test_scores[1],
                "Test AUC-ROC:",
                test_scores[2],
            )
        return {
            "loss": test_scores[0],
            "accuracy": test_scores[1],
            "auc-roc": test_scores[2],
        }

    def get_model(self):
        return self.model
