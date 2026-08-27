import numpy as np
import keras
from keras import layers

size = 64
num_pixels = size * size
kinds = ["raw", "taylor", "fourier"]
activations = ["leaky_relu", "relu", "tanh", "sigmoid"]
output_activations = ["tanh", "leaky_relu", "relu", "sigmoid", "linear"]


def coords():
    # z is the flattened index of a pixel, x and y are its 2d position
    z = np.arange(num_pixels, dtype=np.float32)
    x = z % size
    y = z // size
    return z, x, y


def normalize(vals, max_val, zero_centered, scale):
    if zero_centered:
        v = 2.0 * vals / max_val - 1.0
    else:
        v = vals / max_val
    return scale * v


def make_features(s, z, x, y):
    # s carries kind, order, use_xy, zero_centered and scale for one network
    cols = []
    if s["kind"] == "raw":
        if s["use_xy"]:
            cols.append(normalize(x, size - 1, s["zero_centered"], s["scale"]))
            cols.append(normalize(y, size - 1, s["zero_centered"], s["scale"]))
        else:
            cols.append(normalize(z, num_pixels - 1, s["zero_centered"], s["scale"]))
    elif s["kind"] == "taylor" and s["use_xy"]:
        # the 2 variable taylor series cut off at total degree order
        xn = normalize(x, size - 1, s["zero_centered"], s["scale"])
        yn = normalize(y, size - 1, s["zero_centered"], s["scale"])
        for d in range(s["order"] + 1):
            for a in range(d, -1, -1):
                cols.append((xn ** a) * (yn ** (d - a)))
    elif s["kind"] == "taylor":
        zn = normalize(z, num_pixels - 1, s["zero_centered"], s["scale"])
        for p in range(s["order"] + 1):
            cols.append(zn ** p)
    elif not s["use_xy"]:
        zn = normalize(z, num_pixels - 1, s["zero_centered"], s["scale"])
        for k in range(s["order"]):
            f = 2.0 ** k
            cols.append(np.sin(f * zn))
            cols.append(np.cos(f * zn))
    else:
        xn = normalize(x, size - 1, s["zero_centered"], s["scale"])
        yn = normalize(y, size - 1, s["zero_centered"], s["scale"])
        for k in range(s["order"]):
            f = 2.0 ** k
            cols.append(np.sin(f * xn))
            cols.append(np.cos(f * xn))
            cols.append(np.sin(f * yn))
            cols.append(np.cos(f * yn))
    return np.stack(cols, axis=1).astype(np.float32)


def input_dim(s):
    if s["kind"] == "raw":
        return 2 if s["use_xy"] else 1
    if s["kind"] == "taylor":
        if s["use_xy"]:
            # the terms of total degree up to order, order 4 gives 15
            return (s["order"] + 1) * (s["order"] + 2) // 2
        return s["order"] + 1
    return s["order"] * (4 if s["use_xy"] else 2)


def build_model(s, width, depth, lr):
    # depth is the number of hidden layers, two is enough for a 64x64 image
    model = keras.Sequential(name=s["kind"])
    model.add(layers.Input(shape=(input_dim(s),)))
    for _ in range(depth):
        model.add(layers.Dense(width))
        add_activation(model, s["act"])
    model.add(layers.Dense(1))
    add_activation(model, s["out_act"])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def add_activation(model, act):
    if act == "leaky_relu":
        model.add(layers.LeakyReLU(negative_slope=0.2))
    elif act == "linear":
        model.add(layers.Activation("linear"))
    else:
        model.add(layers.Activation(act))
