__authors__ = ['1747579', '1744604', '1744896']
__group__ = '17'

from utils_data import read_dataset, read_extended_dataset, crop_images
from utils_data import visualize_retrieval

from Kmeans import KMeans, get_colors
from KNN import KNN

import matplotlib.pyplot as plt
import numpy as np
import time

# =====================================================================
# ANÀLISI QUALITATIU
# =====================================================================
def retrieval_by_color(images, color_labels, query_colors):
    """
    Retorna totes les imatges que contenen algun dels colors demanats.
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
    """
    Retorna totes les imatges que coincideixen amb la forma demanada.
    """
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


# =====================================================================
# ANÀLISI QUANTITATIU
# =====================================================================
def get_shape_accuracy(predicted_shapes, gt_shapes):
    """
    Calcula el percentatge d'acert de la classificació de formes.
    """
    predicted_shapes = np.array(predicted_shapes)
    gt_shapes = np.array(gt_shapes)

    correct = np.sum(predicted_shapes == gt_shapes)
    accuracy = correct / len(gt_shapes)

    return accuracy * 100

def get_color_accuracy(predicted_colors, gt_colors):
    """
    Calcula la similitud mitjana (IoU) entre els colors predits i els reals.
    """
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
    """
    Funció base de l'enunciat per extreure estadístiques d'una imatge variant la K.
    Útil per fer un primer estudi visual del mètode 'first'.
    """
    wcds = []
    iterations = []
    times = []

    Ks = range(2, Kmax + 1)

    for k in Ks:
        options = {'km_init': 'first', 'max_iter': 100, 'tolerance': 0}
        km = KMeans(image, K=k, options=options)

        start = time.time()
        km.fit()
        end = time.time()

        wcds.append(km.withinClassDistance())
        iterations.append(km.num_iter)
        times.append(end - start)

    # GRÀFIQUES WCD, ITERACIONS i TEMPS INDIVIDUALS
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

    # Llegim les dades
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # Llista amb totes les classes existents diferents que hi ha.
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Imatges tallades (agafar el que ens interessa).
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    # Tallem les imatges amb les coord donades anteriorment.
    cropped_images = crop_images(imgs, upper, lower)

    # =====================================================================
    # TEST AVALUACIÓ DE KMEANS (init_centroides i bestK)
    # =====================================================================
    print("EVALUACIÓ DE INICIALITZACIONS K-MEANS I BEST-K")
    
    # Seleccionem la 1a imatge
    imatge_proves = train_imgs[0] 
    # Decidim agafar K clústers entre 2 i 7.
    Kmax = 7
    Ks = list(range(2, Kmax + 1))
    metodes_kmeans = ['first', 'random', 'geometric']
    
    # Estructures de dades per emmagatzemar els resultats dels experiments
    wcd_per_metode = {m: [] for m in metodes_kmeans} 
    iter_per_metode = {m: [] for m in metodes_kmeans} 
    temps_per_metode = {m: [] for m in metodes_kmeans} 
    
    # Cada iteració del for executa un mètode diferent
    for metode in metodes_kmeans:
        print(f"Executant anàlisi per al mètode d'inicialització: {metode}...")
        for k in Ks:
            # Volem 100 iteracions i tolindex rànquing_tolerancia 0
            opcions_km = {'km_init': metode, 'max_iter': 100, 'tolerance': 0}
            km = KMeans(imatge_proves, K=k, options=opcions_km)
            
            t_inici = time.time()
            km.fit()
            t_final = time.time()
            
            wcd_per_metode[metode].append(km.withinClassDistance())
            iter_per_metode[metode].append(km.num_iter)
            temps_per_metode[metode].append(t_final - t_inici)

    # GRÀFICA A: Evolució del WCD
    plt.figure(figsize=(8, 5))
    plt.plot(Ks, wcd_per_metode['first'], marker='o', label='Original (first)', linewidth=2)
    plt.plot(Ks, wcd_per_metode['random'], marker='s', label='Millora (random)', linewidth=2)
    plt.plot(Ks, wcd_per_metode['geometric'], marker='^', label='Millora (geometric)', linewidth=2)
    plt.xlabel('Número de Clústers (K)')
    plt.ylabel('Within Class Distance (WCD)')
    plt.title('Comparativa directa del WCD (Anàlisi d\'eficiència BestK)')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()

    # GRÀFICA B: Evolució de la quantitat d'iteracions
    plt.figure(figsize=(8, 5))
    plt.plot(Ks, iter_per_metode['first'], marker='o', label='Original (first)', linewidth=2)
    plt.plot(Ks, iter_per_metode['random'], marker='s', label='Millora (random)', linewidth=2)
    plt.plot(Ks, iter_per_metode['geometric'], marker='^', label='Millora (geometric)', linewidth=2)
    plt.xlabel('Número de Clústers (K)')
    plt.ylabel('Iteracions realitzades fins convergir')
    plt.title('Velocitat d\'optimització: Nombre d\'iteracions vs K')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()


    # =====================================================================
    # AVALUACIÓ DE LES MILLORES: KNN (Estudi de Precisió i Temps de Càlcul)
    # =====================================================================
    print("\n--- EVALUACIÓ DE CONFIGURACIONS KNN ---")
    
    # Definició de les opcions de configuració amb els strings correctes en català
    proves_knn = [
        {'nom': 'Base (Pixel + Euclidean)', 'feat': 'pixel', 'dist': 'euclidean'},
        {'nom': 'Millora 1 (Pixel + Manhattan)', 'feat': 'pixel', 'dist': 'cityblock'},
        {'nom': 'Millora 2 (Reduccio + Euclidean)', 'feat': 'reduccio', 'dist': 'euclidean'},
        {'nom': 'Millora 3 (Descriptors + Euclidean)', 'feat': 'descriptors', 'dist': 'euclidean'}
    ]
    
    noms_grafic = []
    precisions_grafic = []
    temps_grafic = []

    # Llançament dels experiments numèrics del KNN
    for prova in proves_knn:
        opcions_knn = {'knn_features': prova['feat'], 'knn_dist': prova['dist']}
        knn = KNN(train_imgs, train_class_labels, options=opcions_knn)
        
        t_inici = time.time()
        prediccions = knn.predict(test_imgs, k=3)
        t_final = time.time()
        
        temps_execucio = t_final - t_inici
        precisio = get_shape_accuracy(prediccions, test_class_labels)
        
        # Guardem resultats per a la representació gràfica posterior
        noms_grafic.append(prova['nom'])
        precisions_grafic.append(precisio)
        temps_grafic.append(temps_execucio)
        
        print(f"{prova['nom']:<38} -> Precisió: {precisio:.2f}% | Temps de Càlcul: {temps_execucio:.4f}s")

    # GRÀFICA C: Mètrica de Precisió Global de Classificació de Formes
    plt.figure(figsize=(10, 4))
    plt.bar(noms_grafic, precisions_grafic, color='skyblue', edgecolor='black', alpha=0.8)
    plt.ylabel('Precisió d\'encert (%)')
    plt.title('Anàlisi comparatiu de Precisió del Model KNN')
    plt.xticks(rotation=10)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

    # GRÀFICA D: Mètrica del Cost Computacional
    plt.figure(figsize=(10, 4))
    plt.bar(noms_grafic, temps_grafic, color='salmon', edgecolor='black', alpha=0.8)
    plt.ylabel('Temps d\'execució total (Segons)')
    plt.title('Cost Computacional: Temps de Càlcul del KNN segons la millora')
    plt.xticks(rotation=10)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()


    # =====================================================================
    # EXTRACTE DE PROVES QUALITATIVES COMPLEMENTÀRIES
    # =====================================================================
    print("\n--- EXECUTANT PROVA EXTRA DE RECUPERACIÓ PER COLOR ---")
    predicted_colors = []
    for img in test_imgs[:15]:  # Execució limitada a 15 imatges de mostra per estalviar temps
        km = KMeans(img, K=3)
        km.fit()
        colors = get_colors(km.centroids)
        predicted_colors.append(colors)
    
    res = retrieval_by_color(test_imgs[:15], predicted_colors, ['Blue'])
    if len(res) > 0:
        visualize_retrieval(res, min(len(res), 10), info=['Blue'] * len(res), title='Peces detectades com a blaves')
    else:
        print("No s'han trobat peces de roba blaves a la mostra reduïda de test.")
