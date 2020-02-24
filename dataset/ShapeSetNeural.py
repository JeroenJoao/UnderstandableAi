import glob
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from UnderstandableAi.settings import BASE_DIR


def neuralCreate():
    #circles
    images = []
    for img_path in glob.glob('training_set/circles/*.png'):
        images.append(mpimg.imread(img_path))

    plt.figure(figsize=(20,10))
    columns = 5
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)

    #squares
    images = []
    for img_path in glob.glob('training_set/squares/*.png'):
        images.append(mpimg.imread(img_path))

    plt.figure(figsize=(20,10))
    columns = 5
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)

    #triangles
    images = []
    for img_path in glob.glob('shapes/triangles/*.png'):
        images.append(mpimg.imread(img_path))

    plt.figure(figsize=(20,10))
    columns = 5
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)

    #init CNN
    classifier = Sequential()

    classifier.add(Conv2D(32, (3, 3), padding='same', input_shape = (28, 28, 3), activation = 'relu'))
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5)) # antes era 0.25

    # Adding a second convolutional layer
    classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5)) # antes era 0.25​

    # Adding a third convolutional layer
    classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5)) # antes era 0.25

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(units = 512, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 3, activation = 'softmax'))

    #compile
    classifier.compile(optimizer = 'rmsprop',
                       loss = 'categorical_crossentropy',
                       metrics = ['accuracy'])

    #split data
    train_datagen = ImageDataGenerator(rescale = 1./255)
    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('shapes/training_set',
                                                     target_size = (28, 28),
                                                     batch_size = 16,
                                                     class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory('shapes/test_set',
                                                target_size = (28, 28),
                                                batch_size = 16,
                                                class_mode = 'categorical')

    #model checkpoint
    checkpointer = ModelCheckpoint(filepath="best_weights.hdf5",
                                   monitor = 'val_acc',
                                   verbose=1,
                                   save_best_only=True)
    #fir
    history = classifier.fit_generator(training_set,
                                       steps_per_epoch = 100,
                                       epochs = 10,
                                       callbacks=[checkpointer],
                                       validation_data = test_set,
                                       validation_steps = 50)

    #load weights
    #classifier.load_weights('best_weights.hdf5')

    #save model
    #classifier.save('shapes/shapes_cnn.h5')

    json_model = classifier.to_json()
    open('model_architecture.json', 'w').write(json_model)
    classifier.save_weights('model_weights.h5', overwrite=True)


def loadModel():
    path_model = os.path.join(BASE_DIR, 'dataset/ShapeSet/model_architecture.json')
    model = model_from_json(open(path_model).read())
    path_weights = os.path.join(BASE_DIR, 'dataset/ShapeSet/model_weights.h5')
    model.load_weights(path_weights)
    model.compile(optimizer = 'rmsprop',
                       loss = 'categorical_crossentropy',
                       metrics = ['accuracy'])
    return model


def getLayerPlot(img, layerNum):
    model = keras.applications.VGG16(weights='imagenet', include_top = 'False')

    img_path = os.path.join(BASE_DIR,'dataset/ShapeSet/shapes/' + str(img) + '.png')

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    layer_outputs = [layer.output for layer in model.layers][1:] # Extracts the outputs of the top 12 layers
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
    plt.savefig(os.path.join(BASE_DIR,'dataset/ShapeSet/image.png'))


#getLayerPlot('square1', 1)