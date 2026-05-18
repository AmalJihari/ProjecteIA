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

# Analisis quantitatiu
def get_shape_accuracy(predicted_shapes, gt_shapes):
    predicted_shapes = np.array(predicted_shapes)
    gt_shapes = np.array(gt_shapes)

    correct = np.sum(predicted_shapes == gt_shapes)

    accuracy = correct / len(gt_shapes)

    return accuracy * 100

def get_color_accuracy(predicted_colors, gt_colors):
    scores = []
    for pred, gt in zip(predicted_colors, gt_colors):

        pred_set = set(pred)
        gt_set = set(gt)

        intersection = len(pred_set.intersection(gt_set))
        union = len(pred_set.union(gt_set))

        if union == 0:
            score = 0
        else:
            score = intersection / union

        scores.append(score)

    return np.mean(scores) * 100

def Kmean_statistics(image, Kmax):

    wcds = []
    iterations = []
    times = []

    Ks = range(2, Kmax + 1)

    for k in Ks:

        options = { 'km_init': 'first', 'max_iter': 100, 'tolerance': 0}

        km = KMeans(image, K=k, options=options)

        start = time.time()

        km.fit()

        end = time.time()

        wcds.append(km.withinClassDistance())
        iterations.append(km.num_iter)
        times.append(end - start)


    # GRÁFICAS WCD, TIEMPO y It

    plt.figure()

    plt.plot(Ks, wcds, marker='o')

    plt.xlabel('K')
    plt.ylabel('WCD')
    plt.title('WCD vs K')

    plt.show()

    plt.figure()

    plt.plot(Ks, iterations, marker='o')

    plt.xlabel('K')
    plt.ylabel('Iterations')
    plt.title('Iterations vs K')

    plt.show()
    
    plt.figure()

    plt.plot(Ks, times, marker='o')

    plt.xlabel('K')
    plt.ylabel('Time (s)')
    plt.title('Convergence Time vs K')

    plt.show()

    return wcds, iterations, times




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
   
        # You can start coding your functions here
    """
    QUALITATIU - PROVES MARTA
    
    
    Visualitzar imatges originals
    
    res = retrieval_by_color(cropped_images, color_labels, ['Blue'])
    #print(len(res))
    #print(set([c for sub in color_labels for c in sub])) veure els colors (per veure com s'escriuen)
    visualize_retrieval(res, 10, info=['Blue']*len(res), title='Blue items')
    # info lo que fa es mostrar el text de sota les imatges, si ho multipliquem per len(res) cada imatge te un nom
    
    
    Visualitzar imatges retallades (zoom, pero massa pixelades)
    
    
    res = retrieval_by_shape(imgs, class_labels, ['Shirts'])
    print("Shirts:", len(res))
    #print("Formes disponibles:", set(class_labels)) Per eure com s'esciuen les peces de roba
    visualize_retrieval(res, 10, info=['Shirts']*len(res), title='Shirts')

        
     

    
    knn = KNN(train_imgs, train_class_labels)

    predicted_labels = knn.predict(test_imgs, k=3)
    
    res = retrieval_by_shape(
        test_imgs,
        predicted_labels,
        ['Sandals']
    )
    
    visualize_retrieval(
        res,
        12,
        info=['Sandals'] * len(res),
        title='KNN Retrieval - Sandals'
    )
    #print("Formes disponibles:", set(class_labels)) # Per eure com s'esciuen les peces de roba
    res = retrieval_combined(test_imgs, color_labels, class_labels, ['Blue'], ['Shirts'])
    visualize_retrieval(res, 12, info=['Blue Shirts']*len(res), title=' BLue Shirts')

    
    res = retrieval_by_color(test_imgs, color_labels, ['Blue'])
    visualize_retrieval(res, 10, info=['Blue']*len(res), title='Blue items')
"""
    
    """
    KNN: PROVES DE FORMA
    S'utilitza KNN perque el problema de la forma de la roba és un problema
    de classificació supervisada. Tenim:
        - imatges supervisades (train_imgs)
        - sabem quina peça és cada una (train_class_labels)
        - volem predir la categoria de noves imatges
    KNN compara una imatge nova amb les imatges del conjunt d'entrenament i mira quines són les més semblants
    k=3; el classificador mirara els 3 veins més propers
    El valor de k afecta el comportamrnt del model
    """


    """
    KMEANS: PROVES DE COLOR
        agrupa píxels similars segons el seu color RGB.
        Agrupa tots els pixels de la imatge
        busca colors semblants
        crea grups
        cada grup te un centroide (color representatiu), despres els centroides es transformen en etiquetes
        predicted_colors = []

        for img in test_imgs:
        
            km = KMeans(img, K=3)
        
            km.fit()
        
            colors = get_colors(km.centroids)
        
            predicted_colors.append(colors)
        
        res = retrieval_by_color(
            test_imgs,
            predicted_colors,
            ['Blue']
        )
        
        visualize_retrieval(
            res,
            12,
            info=['Blue'] * len(res),
            title='KNN Retrieval - Sandals'
        )
        
    """
    predicted_colors=[]
    for img in imgs:
    
        km = KMeans(img, K=3)
    
        km.fit()
    
        colors = get_colors(km.centroids)
    
        predicted_colors.append(colors)
    
    res = retrieval_by_color(imgs, predicted_colors,['Blue'])
    
    visualize_retrieval(res, 10,info=['Blue'] * len(res),title='Peces Blue')
    
    
    

    
    
    
    
    
    
    
    
    
