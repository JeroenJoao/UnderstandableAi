import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from keras.preprocessing import image
from keras.models import model_from_json
from UnderstandableAi.settings import BASE_DIR
from kerasvismaster.vis.visualization import activation_maximization


def loadModel():
    path_model = os.path.join(BASE_DIR, 'netupload/uploads/model_architecture.json')
    model = model_from_json(open(path_model).read())
    path_weights = os.path.join(BASE_DIR, 'netupload/uploads/model_weights.h5')
    model.load_weights(path_weights)
    model.compile(optimizer = 'rmsprop',
                       loss = 'categorical_crossentropy',
                       metrics = ['accuracy'])
    return model


def getLayerPlot(img, layerNum, size):
    model = loadModel()

    img_path = os.path.join(BASE_DIR, 'dataset/' + img + '.png')

    activation_maximization.visualize_activation(model, 1)

    '''
    
    img = image.load_img(img_path, target_size=(int(size), int(size)))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    layer_outputs = [layer.output for layer in model.layers][1:] # Extracts the outputs of the top 12 layers
    print(layer_outputs)
    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
    activations = activation_model.predict(img_tensor) # Returns a list of five Numpy arrays: one array per layer activation
    first_layer_activation = activations[0]

    plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

    layer_names = []
    for layer in model.layers[:int(layerNum)]:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

    #print(layer_names)
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
    plt.savefig(os.path.join(BASE_DIR,'dataset/CusNet/image.png'))
    plt.close()
    '''

getLayerPlot('uploads/cat', 1, 28)