import numpy as np
from PIL import Image

size = 64


def load_image(path):
    img = Image.open(path)
    img = img.convert("L")
    orig = img.size
    img = center_crop(img)
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    arr = arr / 127.5 - 1.0
    return arr.flatten(), orig


def center_crop(img):
    # take the middle square so the image never gets stretched
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def to_image(values):
    # back from the -1..1 range to a normal 8 bit image
    arr = (np.asarray(values).reshape(size, size) + 1.0) * 127.5
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "L")


def blank_image():
    return Image.new("L", (size, size), 0)
