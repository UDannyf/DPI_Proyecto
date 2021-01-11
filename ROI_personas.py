from cv2 import cv2
from glob import glob
import numpy as np
from matplotlib import pyplot as plt

def segmentacionColor(img):
    image = img
    num_clusters = 10
    # Creamos una copia para poderla manipular a nuestro antojo.
    image_copy = np.copy(image)
 
    # Mostramos la imagen y esperamos que el usuario presione cualquier tecla para continuar.
    #cv2.imshow('Imagen', image)
    #cv2.waitKey(0)
 
    # Convertiremos la imagen en un arreglo de ternas, las cuales representan el valor de cada pixel. En pocas palabras,
    # estamos aplanando la imagen, volviéndola un vector de puntos en un espacio 3D.
    pixel_values = image_copy.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Abajo estamos aplicando K-Means. Como siempre, OpenCV es un poco complicado en su sintaxis, así que vamos por partes.
    
    # Definimos el criterio de terminación del algoritmo. En este caso, terminaremos cuando la última actualización de los
    # centroides sea menor a *epsilon* (cv2.TERM_CRITERIA_EPS), donde epsilon es 1.0 (último elemento de la tupla), o bien
    # cuando se hayan completado 10 iteraciones (segundo elemento de la tupla, criterio cv2.TERM_CRITERIA_MAX_ITER).
    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Este es el número de veces que se correrá K-Means con diferentes inicializaciones. La función retornará los mejores
    # resultados.
    number_of_attempts = 10
    
    # Esta es la estrategia para inicializar los centroides. En este caso, optamos por inicialización aleatoria.
    centroid_initialization_strategy = cv2.KMEANS_RANDOM_CENTERS
    
    # Ejecutamos K-Means con los siguientes parámetros:
    # - El arreglo de pixeles.
    # - K o el número de clusters a hallar.
    # - None indicando que no pasaremos un arreglo opcional de las mejores etiquetas.
    # - Condición de parada.
    # - Número de ejecuciones.
    # - Estrategia de inicialización.
    #
    # El algoritmo retorna las siguientes salidas:
    # - Un arreglo con la distancia de cada punto a su centroide. Aquí lo ignoramos.
    # - Arreglo de etiquetas.
    # - Arreglo de centroides.
    _, labels, centers = cv2.kmeans(pixel_values,num_clusters,
                                    None,
                                stop_criteria,
                                number_of_attempts,
                                centroid_initialization_strategy)
 
    # Aplicamos las etiquetas a los centroides para segmentar los pixeles en su grupo correspondiente.
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    
    # Debemos reestructurar el arreglo de datos segmentados con las dimensiones de la imagen original.
    segmented_image = segmented_data.reshape(image_copy.shape)
    
    # Mostramos la imagen segmentada resultante.
    #cv2.imshow('Imagen segmentada', segmented_image)
    #cv2.waitKey(0)
    return segmented_image

def segmentacionWarershed(img):
    img = img
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # Eliminación del ruido
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # Encuentra el área del fondo
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Encuentra el área del primer
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Encuentra la región desconocida (bordes)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Etiquetado
    ret, markers = cv2.connectedComponents(sure_fg)
    # Adiciona 1 a todas las etiquetas para asegurra que el fondo sea 1 en lugar de cero
    markers = markers+1
    # Ahora se marca la región desconocida con ceros
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    return img

def filtroUnsharping(img):
    gaussian_3 = cv2.GaussianBlur(img, (0, 0), 20)
    img_al = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0)
    return img_al

def filtroSharpen(img):
    sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

    result = cv2.filter2D(img, -1, sharpen)
    return result

def detectFace(image): 
    faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray,
    scaleFactor=1.1,
    minNeighbors=5)
    countFace = 0
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        countFace = countFace +1
    print("conteo de rostros: ",countFace)
    return image
    
def yoloImage(imagen):
    net = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    # Loading image
    #winName = 'Yolo object detection in OpenCV'
    #cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    img = imagen
    #frame_count =0
    #img = cv2.imread("room_ser.jpg")
    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape    
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(indexes)
    count=0
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label=="person":
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 1, color, 1)
                count=count+1
    print('conteo de Personas: ',count)
    return img

################### menu############

path = "images/img1.jpg"
#path = "images/img2.jpg"
#path = "images/img3.jpg"
#path = "images/img4.jpg"
#path = "images/img5.jpg"
while(True):
    print ("\n\n\t0. Salir\n\t1. Cargar una imagen\n\t2. Realzado filtro Unsharping"+
           "\n\t3. Realzado filtro Sharpen \n\t4. Segmentacion Warershed  \n\t5. Segmentacion Color"+
           "\n\t6. Detección rostros\n")
    op = input("\n\tIngrese la opcion --> ")

    if (op=='0'):
        break
    elif (op=='1'):
        print ("\n\tCargar una imagen desde un archivo\n")
        img = cv2.imread(path, 1)
        clon = yoloImage(img)
        cv2.imshow("Imagen", img)
        cv2.imshow("Imagen Yolo", clon)
        cv2.waitKey(0)
        
    elif (op=='2'):
        print ("\n\t Realzado filtro Unsharping\n")
        img = cv2.imread(path,1)
        r1 = filtroUnsharping(img)
        clon = yoloImage(r1)
        cv2.imshow("Imagen", r1)
        cv2.imshow("Imagen Yolo", clon)
        cv2.waitKey(0)
        
    elif (op=='3'):
        print ("\n\tRealzado filtro Sharpen\n")
        img = cv2.imread(path,1)
        r1 = filtroSharpen(img)
        clon = yoloImage(r1)
        cv2.imshow("Imagen", r1)
        cv2.imshow("Imagen Yolo", clon)
        cv2.waitKey(0)
    
    elif (op=='4'):
        print ("\n\tSegmentacion Warershed\n")
        img = cv2.imread(path,1)
        s1 = segmentacionWarershed(img)
        clon = yoloImage(s1)
        cv2.imshow("Imagen Segmentada", s1)
        cv2.imshow("Imagen Yolo", clon)
        cv2.waitKey(0)
    
    elif (op=='5'):
        print ("\n\tSegmentación Color\n")
        img = cv2.imread(path,1)
        s2 = segmentacionColor(img)
        clon = yoloImage(s2)
        cv2.imshow("Imagen Segmentada", s2)
        cv2.imshow("Imagen Yolo", clon)
        cv2.waitKey(0)
    
    elif (op=='6'):
        print ("\n\tDetectar Rostro\n")
        img = cv2.imread(path,1)
        imge = img.copy()
        dt = detectFace(imge)
        cv2.imshow("Imagen original", img)
        cv2.imshow("Conteo rostro", dt)
        cv2.waitKey(0)
        
    else:
        print ("\n\tLa opción no es válida.")     
    cv2.destroyAllWindows()
