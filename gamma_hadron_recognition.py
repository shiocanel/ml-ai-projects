import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

cols = [
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
    "class",
]
df = pd.read_csv("magic04.data", names=cols)
df["class"] = (df["class"] == "g").astype(int)
df.head()

for label in cols[:-1]:
    plt.hist(
        df[df["class"] == 1][label],
        color="blue",
        label="gamma",
        alpha=0.7,
        density=True,
    )
    plt.hist(
        df[df["class"] == 0][label],
        color="red",
        label="hadron",
        alpha=0.7,
        density=True,
    )
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

train, valid, test = np.split(
    df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))]
)


def scale_dataset(df, oversample=False):
    x = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    data = np.hstack((x, np.reshape(y, (-1, 1))))

    return data, x, y


train, X_train, Y_train = scale_dataset(train, oversample=True)
valid, X_valid, Y_valid = scale_dataset(valid, oversample=True)
test, X_test, Y_test = scale_dataset(test, oversample=True)

print(len(train))


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
# it just takes the k nearest neighbors and its making a prediction based on these neighbors

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, Y_train)  # u train dat shii

Y_pred = knn_model.predict(X_test)

print(Y_pred)
print(Y_test)

correct = sum(Y_pred == Y_test)
print(f"accuracy: {correct / len(Y_test) * 100}%")

# there is a better way
print(classification_report(Y_pred, Y_test))


from sklearn.naive_bayes import GaussianNB
# so this uses some sort of formulas that i didnt really quite understood soooooo

nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, Y_train)

Y_pred = nb_model.predict(X_test)
print(classification_report(Y_pred, Y_test))


from sklearn.linear_model import LogisticRegression
# uses the sigmoid function the one that looks like this:

lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, Y_train)

Y_pred = lg_model.predict(X_test)
print(classification_report(Y_pred, Y_test))


from sklearn.svm import SVC
# it basically makes a line trhou our data and make the predictions based on this separation

svm_model = SVC()
svm_model = svm_model.fit(X_train, Y_train)

Y_pred = svm_model.predict(X_test)
print(classification_report(Y_pred, Y_test))


import tensorflow as tf


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"], label="loss")
    ax1.plot(history.history["val_loss"], label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary crossentropy")
    ax1.grid(True)

    ax2.plot(history.history["accuracy"], label="accuracy")
    ax2.plot(history.history["val_accuracy"], label="val_accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)

    plt.show()


def train_model(
    X_train, Y_train, num_nodes, dropout_prob, learn_rate, batch_size, epochs
):
    nn_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(num_nodes, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dropout(dropout_prob),  # to prevent overfitting
            tf.keras.layers.Dense(num_nodes, activation="relu"),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learn_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # train the model
    history = nn_model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
    )

    return nn_model, history


min_val_loss = float("inf")
min_model = None

epochs = 100
for num_nodes in [8, 16]:
    for dropout_prob in [0, 0.2]:
        for learn_rate in [0.001, 0.005, 0.01]:
            for batch_size in [32, 64]:
                print(
                    f"nodes: {num_nodes} dropout probability: {dropout_prob} learning rate: {learn_rate} batch size: {batch_size}"
                )
                model, history = train_model(
                    X_train,
                    Y_train,
                    num_nodes,
                    dropout_prob,
                    learn_rate,
                    batch_size,
                    epochs,
                )
                plot_history(history)
                val_loss = model.evaluate(X_valid, Y_valid)[0]

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    min_model = model

Y_pred = min_model.predict(X_test)
Y_pred = (
    (Y_pred > 0.5)
    .astype(int)
    .reshape(
        -1,
    )
)

print(classification_report(Y_pred, Y_test))
