import os
from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np
from keras import backend as K

from keras.preprocessing import image

from UnderstandableAi.settings import BASE_DIR

def getLayerPlot(img, layer):
    K.clear_session()
    model = keras.applications.VGG16(weights='imagenet', include_top = 'False')


    img_path = os.path.join(BASE_DIR,
                            'dataset/ResNetSet/' + str(img) + '.png')

    img = image.load_img(img_path, target_size=(224, 224))

    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    layer_outputs = [layer.output for layer in model.layers][1:]

    activation_model = keras.models.Model(inputs=model.input,
                                    outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input

    activations = activation_model.predict(
        img_tensor)  # Returns a list of five Numpy arrays: one array per layer activation

    first_layer_activation = activations[0]
    plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
    layer_names = []
    for layer in model.layers[:int(layer)]:
        layer_names.append(layer.name)
    images_per_row = 8
    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.savefig(os.path.join(BASE_DIR, 'dataset/ResNetSet/image.png'))
    K.clear_session()
