import os
import sys
from cv2 import cv2
from glob import glob
import numpy as np
from matplotlib import pyplot as plt

def s1(img): #segmentacionColor
    image = img
    num_clusters = 100
    # Creamos una copia para poderla manipular a nuestro antojo.
    image_copy = np.copy(image)
    pixel_values = image_copy.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    number_of_attempts = 10
    
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
    return segmented_image

def s2(img): #segmentacionWarershed
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

def r1(img): #filtroUnsharping
    gaussian_3 = cv2.GaussianBlur(img, (0, 0), 20)
    img_al = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0)
    return img_al

def r2(img): #filtroSharpen
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
<<<<<<< HEAD
    #print("conteo de rostros: ",countFace)
    return image, countFace
    
def yolo(imagen):
=======
    print("conteo de rostros: ",countFace)
    return image
    
def yoloImage(imagen):
>>>>>>> e58be58d019452416e08f2c40c0d60a1eb2895e6
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
    #print('conteo de Personas: ',count)
    return img, count

<<<<<<< HEAD
################### Combinacion segmentacion-realzado #################

def s1r1(image):    
    img1 = s1(image)
    img2 = r1(img1)
    img, count = yolo(img2)
    img, face = detectFace(img)    
    return img, count, face
=======
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
>>>>>>> e58be58d019452416e08f2c40c0d60a1eb2895e6

def s1r2(image):
    img1 = s1(image)
    img2 = r2(img1)
    img, count = yolo(img2)
    img, face = detectFace(img)    
    return img, count, face

def s2r1(image):
    img1 = s2(image)
    img2 = r1(img1)
    img, count = yolo(img2)    
    img, face = detectFace(img)    
    return img, count, face

def s2r2(image):
    img1 = s2(image)
    img2 = r2(img1)
    img, count = yolo(img2)    
    img, face = detectFace(img)    
    return img, count, face

def mayor(num1, num2, num3, num4):
    arreglo = np.array([num1, num2, num3, num4])
    maxElement = np.amin(arreglo)
    i1 = 0
    i2 = 0
    i3 = 0
    i4 = 0
    if maxElement == num1:
        i1 = 1
    if maxElement == num2:
        i2 = 1
    if maxElement == num3:
        i3 = 1
    if maxElement == num4:
        i4 = 1
    return i1, i2, i3, i4
    
<<<<<<< HEAD
            


plistx = ["seg1_real_1","seg1_real2","seg2_real1","seg2_real2"]
plisty = [0,0,0,0]
rlistx = ["seg1_real_1","seg1_real2","seg2_real1","seg2_real2"]
rlisty = [0,0,0,0]

for path in glob('img_prueba/*.jpg'):
    os.system("cls")
    img = cv2.imread(path, 1)    
    cv2.imshow("Imagen", img)
    cv2.waitKey(0)
    pr0 = input("\n\tIngrese # personas --> ")
    fc0 = input("\n\tIngrese # rotros --> ")
    p0 = int(pr0)
    f0 = int(fc0)
    if (p0 == -1):
        break
    img1, p1, f1 = s1r1(img)
    img2, p2, f2 = s1r2(img)
    img3, p3, f3 = s2r1(img)
    img4, p4, f4 = s2r2(img)
    ep1 = 100 -(p1*100/p0)
    ep2 = 100 -(p2*100/p0)
    ep3 = 100 -(p3*100/p0)
    ep4 = 100 -(p4*100/p0)
    er1 = 100 -(f1*100/f0)
    er2 = 100 -(f2*100/f0)
    er3 = 100 -(f3*100/f0)
    er4 = 100 -(f4*100/f0)
    i0,i1,i2,i3 = mayor(ep1,ep2,ep3,ep4)
    x0,x1,x2,x3 = mayor(er1,er2,er3,er4)
    plisty[0] +=i0
    plisty[1] +=i1
    plisty[2] +=i2
    plisty[3] +=i3
    rlisty[0] +=x0
    rlisty[1] +=x1
    rlisty[2] +=x2
    rlisty[3] +=x3
    #cv2.imshow('Segmentación1_realzado1', img1)
    #cv2.imshow('Segmentación1_realzado2', img1)
    #cv2.imshow('Segmentación2_realzado1', img1)
    #cv2.imshow('Segmentación2_realzado2', img1)
    #concat_horizontal = cv2.hconcat([img1, img2])
    #concat_horizontal1 = cv2.hconcat([img3, img4])
    #cv2.imshow('concat_horizontal', concat_horizontal)
    #cv2.imshow('concat_horizontal', concat_horizontal1)
    print("Datos      |  GT  | Segmenta1_realza1 | Segmenta1_realza2 | Segmenta2_realza1 | Segmenta2_realza2")
    print("Personas   |  %d  |      %d           |         %d        |        %d         |         %d       " % (p0, p1, p2, p3, p4))
    print("Err_Per    | 0.0  |      %.2f         |         %.2f      |        %.2f       |         %.2f       " % (ep1, ep2, ep3, ep4))
    print("Rostros    |  %d  |      %d           |         %d        |        %d         |         %d       " % (f0, f1, f2, f3, f4))
    print("Err_rost   | 0.0  |      %.2f         |         %.2f      |        %.2f       |         %.2f       " % (er1, er2, er3, er4))    
    cv2.waitKey(0)
=======
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
>>>>>>> e58be58d019452416e08f2c40c0d60a1eb2895e6
    cv2.destroyAllWindows()

plt.bar(plistx,plisty,align = "center")
plt.title("Mejor rendimiento para Detectar personas")
plt.xlabel("Algoritmos")
plt.ylabel("Imagenes")
plt.show()

plt.bar(rlistx, rlisty, align = "center")
plt.title("Mejor rendimiento para detectar Rostros")
plt.xlabel("Algoritmos")
plt.ylabel("Imagenes")
plt.show()