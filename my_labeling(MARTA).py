__authors__ = ['1747579', '1744604', '1744896']
__group__ = '17'

from utils_data import read_dataset, read_extended_dataset, crop_images
from utils_data import visualize_retrieval

from Kmeans import KMeans
from KNN import KNN

import numpy as np

# Analisis qualitatiu
def retrieval_by_color(images, color_labels, query_colors):
    """
    Retorna totes les images que contene algun dels colors demanats.
    images: llista d'imatges
    color_labels: llista de llistes de colors per imatge
    query_colors: llista de colors a buscar
    """
    res = []
    for img, labels in zip(images, color_labels):
        if any(x in labels for x in query_colors):
            res.append(img)
    return res

def retrieval_by_shape(images, shape_labels, query_shapes):
    res = []
    for img, labels in zip(images, shape_labels):
        if labels in query_shapes:
            res.append(img)
    return res

def retrieval_combined(images, color_labels, shape_labels, query_colors, query_shapes):
    res = []
    for img, color, shape in zip(images, color_labels, shape_labels):
        if any(x in color for x in query_colors) and shape in query_shapes:
            res.append(img)
    return res


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    """
    QUALITATIU
    
    
    Visualitzar imatges originals
    
    res = retrieval_by_color(cropped_images, color_labels, ['Blue'])
    #print(len(res))
    #print(set([c for sub in color_labels for c in sub])) veure els colors (per veure com s'escriuen)
    visualize_retrieval(res, 10, info=['Blue']*len(res), title='Blue items')
    # info lo que fa es mostrar el text de sota les imatges, si ho multipliquem per len(res) cada imatge te un nom
    
    
    Visualitzar imatges retallades (zoom, pero massa pixelades)
    
    res = retrieval_by_color(imgs, color_labels, ['Blue'])
    visualize_retrieval(res, 10, info=['Blue']*len(res), title='Blue items')
    """
    res = retrieval_by_shape(imgs, class_labels, ['Shirts'])
    print("Shirts:", len(res))
    #print("Formes disponibles:", set(class_labels)) Per eure com s'esciuen les peces de roba
    visualize_retrieval(res, 10, info=['Shirts']*len(res), title='Shirts')

    
    
    
    
    
    
    
    
    
    
