import cv2
import numpy as np
from yolo_model import YOLO

#literatur parametreleri
yolo = YOLO(0.6, 0.5)

#etiketler icin txt
file = "data/coco_classes.txt"


#coco_classes icersindeki etiketleri alalim
with open(file) as f:
    class_name = f.readlines()
    
#bosluk silme strip 
all_classes = [c.strip() for c in class_name]

#gorselimiz
f = "dog_cat.jpg"
path = "images/"+f
image = cv2.imread(path)
cv2.imshow("image",image)

#egitecegimiz netural network girdisi 416x416 dir bu nedenle resize ediuoruz
pimage = cv2.resize(image, (416,416))
pimage = np.array(pimage, dtype = "float32")
pimage /= 255.0 #normalizasyon yapiyoruz
#genişletiliyor, axis = 0 demek bir tane daha satır ekliyoruz cunku yolo için gerekli olan bir sey
pimage = np.expand_dims(pimage, axis = 0)

# yolo icin gerekli olan etiketler
# kutu, etiket, dogruluk
boxes, classes, scores = yolo.predict(pimage, image.shape)

#kutu icerisine almak icin for dongusu kuruyoruz
for box, score, cl in zip(boxes, scores, classes):
    
    x,y,w,h = box
    
    #kutunun +0.5 diyerek pay bırakıyoruz kutu daha guzel gozuksun diye
    top = max(0, np.floor(x + 0.5).astype(int))
    left = max(0, np.floor(y + 0.5).astype(int))
    right = max(0, np.floor(x + w + 0.5).astype(int))
    bottom = max(0, np.floor(y + h + 0.5).astype(int))

    #ciziyoruz
    cv2.rectangle(image, (top,left), (right, bottom),(255,0,0),2)
    #yaziyoruz
    cv2.putText(image, "{} {}".format(all_classes[cl],score),(top,left-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1,cv2.LINE_AA)

#gosteriyoruz 
cv2.imshow("yolo",image)    

k = cv2.waitKey(0) & 0xFF #esc tusu

if k == 27: 
    cv2.destroyAllWindows()





























