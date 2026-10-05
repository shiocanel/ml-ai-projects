import numpy as np
import pandas as pd
from matplotlib import pyplot as plt # is for image rendering

import tkinter as tk
from PIL import Image, ImageGrab, ImageOps

import time

data = pd.read_csv("C:\\Users\\Admin\\Downloads\\train\\train.csv")
final_accuracy = 0

print(data.head()) # [42000 rows x 785 columns]

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# print(m, n) # 42000 tests    785 pixels

data_dev = data[0:1000].T # the dev is used to test it on unseen drawings
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255

data_train = data[1000:m].T # train data
Y_train = data_train[0] # label
X_train = data_train[1:n] # pixels
X_train = X_train / 255

# so i understand that if i just imlplement the math it works but i should also understand it

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    B1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    B2 = np.random.rand(10, 1) - 0.5
    return W1, B1, W2, B2

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def back_prop(W1, Z1, A1, W2, Z2, A2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    dB2 = (1 / m) * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    dB1 = (1 / m) * np.sum(dZ1)
    return dW1, dB1, dW2, dB2

def update_params(W1, B1, dW1, dB1, W2, B2, dW2, dB2, alfa):
    W2 = W2 - alfa * dW2
    B2 = B2 - alfa * dB2
    W1 = W1 - alfa * dW1
    B1 = B1 - alfa * dB1
    return W1, B1, W2, B2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def train(X, Y, alfa, iterations):
    W1, B1, W2, B2 = init_params()

    for i in range(iterations + 1):
        Z1, A1, Z2, A2 = forward_prop(W1, B1, W2, B2, X)
        dW1, dB1, dW2, dB2 = back_prop(W1, Z1, A1, W2, Z2, A2, X, Y)
        W1, B1, W2, B2 = update_params(W1, B1, dW1, dB1, W2, B2, dW2, dB2, alfa)


        if i % 10 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)

            print(f"accuracy: {accuracy}")


    return W1, B1, W2, B2, accuracy


W1, B1, W2, B2, final_accuracy = train(X_train, Y_train, 0.10, 500)




def print_image(index, X, Y):
    print(f"label: {Y[index]}")
    for i in range(0, 784):
        print("+" if X[i, index] != 0 else "_", end = " ")
        if (i + 1) % 28 == 0:
            print()

    print()
    print()

def test_prediction(index, W1, B1, W2, B2, X, Y):
    current_image = X[ : , index, None]

    _, _, _, A2 = forward_prop(W1, B1, W2, B2, X[ : , index, None])

    print(f"prediction: {get_predictions(A2)}")
    print_image(index, X, Y) # i dont know meeeeen

    for i in range(10):
        print(f"{i}: {A2[i, 0]}      {"X" if A2[i, 0] == np.max(A2) else ""}")




    plt.axis('off')
    pred_text = get_predictions(A2) 
    plt.text(0, -2, f"Label: {Y[index]}    Prediction: {pred_text}", 
             color="black", fontsize=12, backgroundcolor="white")

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation="nearest")
    plt.show()

def wait():
    print()
    time.sleep(1)
    print("wait...")
    time.sleep(5)
    print()
    print()


def unseen_tests(X, Y, W1, B1, W2, B2):
    _, _, _, A2 = forward_prop(W1, B1, W2, B2, X)

    predictions = get_predictions(A2)
    accuracy = get_accuracy(predictions, Y)

    print(f"accuracy on unseen tests: {accuracy}")

    wait()

unseen_tests(X_dev, Y_dev, W1, B1, W2, B2)


print(f"final accuracy: {final_accuracy}")

index = 2
# while 1:
#     index = int(input("index: "))

#     if index == -1:
#         break

#     test_prediction(index, W1, B1, W2, B2, X_train, Y_train)
#     print(f"final accuracy: {final_accuracy}")

#     wait()


from PIL import Image, ImageDraw, ImageOps

CANVAS_SIZE = 280
IMG_SIZE = 28

# Create a blank image to draw on
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)  # black background
draw_pil = ImageDraw.Draw(image)



def draw(event):
    x, y = event.x, event.y
    r = 8  # stroke size
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
    draw_pil.ellipse([x-r, y-r, x+r, y+r], fill=255)  # draw on Pillow image too

def clear():
    canvas.delete("all")
    draw_pil.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)

def predict():
    # Take Pillow image, resize and normalize
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = ImageOps.invert(img)  # make background white, digit black
    img_data = np.array(img) / 255.0
    img_data = img_data.reshape(IMG_SIZE*IMG_SIZE, 1)
    
    _, _, _, A2 = forward_prop(W1, B1, W2, B2, img_data)
    pred = get_predictions(A2)
    print(f"Prediction: {pred[0]}")

root = tk.Tk()
root.title("Draw a digit")

canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='black')
canvas.pack()
canvas.bind("<B1-Motion>", draw)

btn_frame = tk.Frame(root)
btn_frame.pack()
tk.Button(btn_frame, text="Predict", command=predict).pack(side=tk.LEFT)
tk.Button(btn_frame, text="Clear", command=clear).pack(side=tk.LEFT)

root.mainloop()



print("ok")