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

def getSaliency(img):
    K.clear_session()

    model = VGG16(weights='imagenet', include_top = 'False')
    
    CLASS_INDEX = json.load(open(os.path.join(BASE_DIR, "dataset/ResNetSet/imagenet_class_index.json")))
    classlabel = []
    for i_dict in range(len(CLASS_INDEX)):
        classlabel.append(CLASS_INDEX[str(i_dict)][1])
    _img = load_img(os.path.join(BASE_DIR, 'dataset/ResNetSet/' + img + '.png'), target_size=(224, 224))
    img = img_to_array(_img)
    img = preprocess_input(img)
    y_pred = model.predict(img[np.newaxis, ...])
    class_idxs_sorted = np.argsort(y_pred.flatten())[::-1]
    #topNclass = 5
    #for i, idx in enumerate(class_idxs_sorted[:topNclass]):
    #    print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(
    #        i + 1, classlabel[idx], idx, y_pred[0, idx]))


    layer_idx = utils.find_layer_idx(model, 'predictions')

    model.layers[layer_idx].activation = keras.activations.linear
    #model = utils.apply_modifications(model)


    class_idx = class_idxs_sorted[0]


    grad_top1 = visualize_saliency(model,
                                   layer_idx,
                                   filter_indices=class_idx,
                                   seed_input=img[np.newaxis, ...],
                                   backprop_modifier='guided')

    grad_top2 = visualize_cam(model,
                                   layer_idx,
                                   filter_indices=class_idx,
                                   seed_input=img[np.newaxis, ...],
                                   backprop_modifier='guided')

    def plot_map(grads, grads2):
        fig, axes = plt.subplots(2, 1, figsize=(7, 14))
        axes[0].imshow(_img)
        i = axes[0].imshow(grads, cmap="jet", alpha=0.8)
        fig.colorbar(i)
        axes[1].imshow(_img)
        j = axes[1].imshow(grads2, cmap="jet", alpha=0.8)
        #fig.colorbar(j)
        plt.suptitle("Pr(class={}) = {:5.2f}".format(
            classlabel[class_idx],
            y_pred[0, class_idx]))
        #plt.show()
        plt.savefig(os.path.join(BASE_DIR, 'dataset/ResNetSet/image2.png'))
        plt.close()
    plot_map(grad_top1, grad_top2)
    K.clear_session()


#getSaliency('a')