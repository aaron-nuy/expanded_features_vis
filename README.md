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

- raw index: just the position as a number, either the flat index or x and y
- taylor features: the position as a power series 1, z, z^2, z^3 ... and in
  two variables with cross terms when the x and y box is checked
- fourier features: sin and cos of the position at frequencies 1, 2, 4, 8 ...
  either from the flat index or from the x and y coordinates

start training and watch which representation picks up shape and detail
first. everything runs locally on cpu with keras

run with python app.py (needs keras, numpy, pillow, matplotlib and tkinter)
