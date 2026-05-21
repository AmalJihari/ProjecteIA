__authors__ = ['1747579', '1744604', '1744896']
__group__ = '17'

from utils_data import read_dataset, read_extended_dataset, crop_images
from utils_data import visualize_retrieval

from Kmeans import KMeans, get_colors
from KNN import KNN

import matplotlib.pyplot as plt
import numpy as np
import time



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
    
    
    
    
    
    # ========== EJECUCIÓN GET_SHAPE_ACCURACY
    print("\n" + "="*60)
    print("ANÁLISIS DE PRECISIÓN DEL CLASIFICADOR KNN (SHAPE ACCURACY)")
    print("="*60)
    
    # Entrenar el KNN con los datos de entrenamiento
    knn = KNN(train_imgs, train_class_labels)
    
    # Probar diferentes valores de k
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    precisiones = []
    
    print(f"\n{'k':<4} {'Predicciones correctas':<22} {'Total imágenes':<15} {'Precisión (%)':<15}")
    print("-" * 60)
    
    for k in k_values:
        # Predecir las formas del conjunto de test
        predicted_shapes = knn.predict(test_imgs, k)
        
        # Calcular precisión
        accuracy = get_shape_accuracy(predicted_shapes, test_class_labels)
        precisiones.append(accuracy)
        
        # Calcular número de aciertos
        aciertos = np.sum(predicted_shapes == np.array(test_class_labels))
        
        print(f"{k:<4} {aciertos:<22} {len(test_class_labels):<15} {accuracy:<15.2f}")
    
    # Encontrar el mejor k
    mejor_k = k_values[np.argmax(precisiones)]
    mejor_precision = max(precisiones)
    

    
    # ========== Gráfica    

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, precisiones, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Valor de k (número de vecinos)', fontsize=12)
    plt.ylabel('Precisión (%)', fontsize=12)
    plt.title('Evolución de la precisión del KNN en función de k', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    # Marcar el mejor k
    plt.plot(mejor_k, mejor_precision, 'ro', markersize=12, label=f'Mejor k={mejor_k}')
    plt.legend()
    plt.show()
    

    # ========== EJECUCIÓN GET_COLOR_ACCURACY
    
    print("\n" + "="*60)
    print("ANÁLISIS DE PRECISIÓN DEL CLASIFICADOR K-MEANS (COLOR ACCURACY)")
    print("="*60)
    
    # Probar diferentes valores de K
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    precisiones_color = []
    
    print(f"\n{'K':<4} {'Jaccard medio (%)':<20} {'Observaciones':<30}")
    print("-" * 60)
    
    for k in k_values:
        colores_predichos_por_imagen = []
        
        for i, img in enumerate(cropped_images):
            # Configurar y ejecutar K-means
            options = {'km_init': 'first', 'max_iter': 100, 'tolerance': 0}
            km = KMeans(img, K=k, options=options)
            km.fit()
            
            # Obtener colores de los centroides
            colores_img = get_colors(km.centroids)
            colores_predichos_por_imagen.append(colores_img)
        
        # Calcular precisión media con índice de Jaccard
        accuracy = get_color_accuracy(colores_predichos_por_imagen, color_labels)
        precisiones_color.append(accuracy)
        
        print(f"{k:<4} {accuracy:<20.2f} {'':<30}")
    
    # Encontrar el mejor K
    mejor_k = k_values[np.argmax(precisiones_color)]
    mejor_precision = max(precisiones_color)
    
    
    # ========== Gráfica 
    
    # Gráfica de precisión vs K
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, precisiones_color, 'go-', linewidth=2, markersize=8)
    plt.xlabel('K (número de clusters)', fontsize=12)
    plt.ylabel('Índice de Jaccard medio (%)', fontsize=12)
    plt.title('Evolución de la precisión del color en función de K', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    # Marcar el mejor K
    plt.plot(mejor_k, mejor_precision, 'ro', markersize=12, label=f'Mejor K={mejor_k}')
    plt.legend()
    plt.show()
