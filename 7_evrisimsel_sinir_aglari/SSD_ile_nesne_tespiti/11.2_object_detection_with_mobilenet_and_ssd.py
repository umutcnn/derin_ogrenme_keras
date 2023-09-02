import numpy as np
import os
import cv2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0,255,size=(len(CLASSES),3))

#dosyalarimin yollarini aliyorum
prototxtPath = r"C:\Users\UMUT\python\openCV\OpenCV_3\OpenCV_3\7_evrisimsel_sinir_aglari\SSD_ile_nesne_tespiti\MobileNetSSD_deploy.prototxt.txt"
weightsPath = r"C:\Users\UMUT\python\openCV\OpenCV_3\OpenCV_3\7_evrisimsel_sinir_aglari\SSD_ile_nesne_tespiti\MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)


files =os.listdir()
img_path_list = []

#jpg veya png olanları al
for f in files:
    if f.endswith(".jpg") or f.endswith(".png"): 
        img_path_list.append(f)

print(img_path_list)


for i in img_path_list:
    
    image = cv2.imread(i)

    #yukseklik ve genislik degerlerini aliyorum
    (h,w) = image.shape[:2]

    #resimi modelimize uygun hale getiriyoruz.
    # 300x300 resmin boyutu 
    #0.007843 görüntü piksellerinin normalize edilmesi için kullanılan olcek faktorudur
    #300x300 pencerenin bekledigi boyut
    #127.5 piksel değerlerinin ortalanması için kullanılan ortalama değeri temsil eder, piksel değerleri bu değerden çıkarılarak -1 ile 1 arasında ölçeklenir
    blob = cv2.dnn.blobFromImage(cv2.resize(image,(300, 300)), 0.007843,(300, 300), 127.5)
    
    net.setInput(blob)
    detections = net.forward()
    
    #gorsellestirme islemini yapiyoruz
    for j in np.arange(0, detections.shape[2]):
        
        confidence = detections[0,0,j,2]
        
        if confidence > 0.10:
            
            idx = int(detections[0,0,j,1])
            box = detections[0,0,j,3:7]*np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            
            label = "{}: {}".format(CLASSES[idx], confidence)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx],2)
            y = startY - 16 if startY -16 >15 else startY + 16
            cv2.putText(image, label, (startX,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5,COLORS[idx],2)
            
    cv2.imshow("ssd",image)
    if cv2.waitKey(0) & 0xFF == ord("q"): continue
    
