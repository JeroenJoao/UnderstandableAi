from keras.engine.saving import model_from_json
from tensorflow import keras
from keras.applications.vgg16 import VGG16, preprocess_input
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from keras.preprocessing.image import load_img, img_to_array
from kerasvismaster.vis.visualization import visualize_saliency, visualize_activation
from kerasvismaster.vis.utils import utils
import os
import numpy as np
from UnderstandableAi.settings import BASE_DIR
from keras import backend as K
from kerasvismaster.vis.visualization import visualize_cam


def loadModel(path_model):
    #path_model = os.path.join(BASE_DIR, 'dataset/ShapeSet/model_architecture.json')
    model = model_from_json(open(path_model).read())
    path_weights = os.path.join(BASE_DIR, 'dataset/ShapeSet/model_weights.h5')
    model.load_weights(path_weights)
    model.compile(optimizer = 'rmsprop',
                       loss = 'categorical_crossentropy',
                       metrics = ['accuracy'])
    return model


def getSaliency(img, path_model, size, layer_idx_grad1, layer_idx_grad2):
    K.clear_session()

    model = loadModel(path_model)

    CLASS_INDEX = json.load(open(os.path.join(BASE_DIR, "dataset/ShapeSet/classes.json")))
    classlabel = []
    for i_dict in range(len(CLASS_INDEX)):
        classlabel.append(CLASS_INDEX[str(i_dict)])

    _img = load_img(os.path.join(BASE_DIR, 'dataset/ShapeSet/shapes/' + img + '.png'), target_size=(size, size))
    img = img_to_array(_img)
    img = preprocess_input(img)
    y_pred = model.predict(img[np.newaxis, ...])
    class_idxs_sorted = np.argsort(y_pred.flatten())[::-1]
    # topNclass = 5
    # for i, idx in enumerate(class_idxs_sorted[:topNclass]):
    #    print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(
    #        i + 1, classlabel[idx], idx, y_pred[0, idx]))


    model.layers[layer_idx_grad1].activation = keras.activations.linear
    model.layers[layer_idx_grad2].activation = keras.activations.linear

    # model = utils.apply_modifications(model)

    class_idx = class_idxs_sorted[0]

    grad_top1 = visualize_saliency(model,
                                   layer_idx_grad1,
                                   filter_indices=class_idx,
                                   seed_input=img[np.newaxis, ...],
                                   backprop_modifier='guided')

    grad_top2 = visualize_cam(model,
                              layer_idx_grad2,
                              filter_indices=class_idx,
                              seed_input=img[np.newaxis, ...],
                              backprop_modifier='guided')

    def plot_map(grads, grads2):
        plt.figure(figsize=(8, 12))
        plt.subplot(211)
        plt.imshow(_img)
        i = plt.imshow(grads, cmap="jet", alpha=0.8)
        plt.colorbar(i)
        plt.subplot(212)
        plt.imshow(_img)
        j = plt.imshow(grads2, cmap="jet", alpha=0.8)
        plt.colorbar(j)
        plt.suptitle("Pr(class={}) = {:5.2f}".format(
            classlabel[class_idx],
            y_pred[0, class_idx]), fontsize=20)
        # plt.show()
        plt.savefig(os.path.join(BASE_DIR, 'dataset/ResNetSet/image2.png'))
        plt.close()

    plot_map(grad_top1, grad_top2)
    K.clear_session()

getSaliency('circle2', 28, 15, 8)