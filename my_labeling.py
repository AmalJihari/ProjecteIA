__authors__ = ['1747579', '1744604', '1744896']
__group__ = '14'

import numpy as np
import utils

from utils_data import read_dataset, read_extended_dataset, crop_images, read_one_img
from KNN import KNN


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs_grayscale, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json', with_color=False)

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    knn=KNN(train_imgs_grayscale, train_class_labels)
    
    
    knn.get_k_neighbours([train_imgs_grayscale[3:4]], 7)
    
    print(knn.neighbors)
    
    #PROVA AMB IMATGE EXTERNA: 
    ruta='./images/test/2251.jpg'
    test_img=read_one_img(ruta, w=60, h=80, with_color=False)
    knn.get_k_neighbours(np.array([test_img]), k=7)
    print(knn.neighbors)
    
    
    
    
    
    
    
    
