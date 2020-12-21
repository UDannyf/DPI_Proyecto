from cv2 import cv2
from glob import glob
import numpy as np

def filtroUnsharping(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    img_fil = blur / 150
    img_res = img - img_fil
    img_al = cv2.equalizeHist(img_res)
    return img_al

def filtroHomomorfico(img):
    img = np.float32(img)
    img = img/255

    rows,cols,dim=img.shape
    rh, rl, cutoff = 2.5,0.5,32
    imgYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y,cr,cb = cv2.split(imgYCrCb)

    y_log = np.log(y+0.01)
    y_fft = np.fft.fft2(y_log)
    y_fft_shift = np.fft.fftshift(y_fft)

    DX = cols/cutoff
    G = np.ones((rows,cols))
    for i in range(rows):
        for j in range(cols):
            G[i][j]=((rh-rl)*(1-np.exp(-((i-rows/2)**2+(j-cols/2)**2)/(2*DX**2))))+rl

    result_filter = G * y_fft_shift
    result_interm = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter)))
    result = np.exp(result_interm)
    return result

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
    img = cv2.resize(img, None, fx=2, fy=2)
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

################### munu############

path = "images/img1.jpg"
#path = "images/img2.jpg"
#path = "images/img3.jpg"
while(True):
    print ("\n\n\t0. Salir\n\t1. Cargar una imagen\n\t2. Realzado filtro Unsharping"+
           "\n\t3. Realzado filtroHomomorfico \n")
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
        print ("\n\tRealzado filtroHomomorfico\n")
        img = cv2.imread(path,1)
        r1 = filtroHomomorfico(img)
        clon = yoloImage(r1)
        cv2.imshow("Imagen", r1)
        cv2.imshow("Imagen Yolo", clon)
        cv2.waitKey(0)
   
    else:
        print ("\n\tLa opción no es válida.")     
    cv2.destroyAllWindows()
