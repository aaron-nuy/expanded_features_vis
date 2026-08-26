# expanded_features_vis

a prototype application that allows visualizing and exploring feature mapping
in neural networks to gain better intuition and understanding

you upload an image, it gets center cropped and squeezed to 64x64 grayscale,
then three small mlp networks try to learn it at the same time. each network
only gets information about where a pixel is and has to predict how bright
that pixel should be, so after training the whole picture lives inside the
network weights

the three networks differ only in how the pixel position is expanded into
input features

- raw index: just the flattened position (y * width + x) as a single number
- taylor features: the position expanded as a power series 1, z, z^2, z^3 ...
- fourier features: sin and cos of the position at frequencies 1, 2, 4, 8 ...
  optionally taken from the x and y coordinates instead of the flat index

start training and watch which representation picks up shape and detail
first. everything runs locally on cpu with keras

run with python app.py (needs keras, numpy, pillow, matplotlib and tkinter)
