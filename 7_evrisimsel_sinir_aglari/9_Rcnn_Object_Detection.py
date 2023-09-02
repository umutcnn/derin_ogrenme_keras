import cv2
import numpy as np
import random
from keras.preprocessing.image import img_to_array
from keras.models import load_model


#gorseli alma
image = cv2.imread("mnist.png")
cv2.imshow("Image", image)

#ilklendir yani SelectiveSearch yap
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality()

print("ss")
rects = ss.process()

#kutucuklar ve olasılıklar icin bos liste olusturuyorum
proposals = []
boxes = []
output = image.copy()

for (x,y,w,h) in rects[:25]:
    #her kutu icin rastgele renk uretiyoruz
    color = [random.randint(0,255) for j in range(0,3)]
    #sonra bunlari ciziyorum
    cv2.rectangle(output, (x,y), (x+w,y+h), color, 2)
    
    #Region of Interest yani ilgilenilen bolge
    roi = image[y : y+h, x : x + w]

    #resize yapıyoruz algoritmaya uygun gorselleti olusturmak icin, 
    #interpolation = cv2.INTER_LANCZOS4 goruntu kalitesini arttirmak icin kullaniyoruz
    roi = cv2.resize(roi, dsize = (32,32), interpolation = cv2.INTER_LANCZOS4)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    roi = img_to_array(roi)
    proposals.append(roi)
    boxes.append((x, y, x + w, y + h))
    
proposals = np.array(proposals, dtype = "float64")
boxes = np.array(boxes, dtype = "int32")

#modelimizi yukluyoruz
print("siniflandirma")
model = load_model("modelWeights.h5")
proba = model.predict(proposals)

number_list = []
idx = []

#sırayla dolasiyorum en yuksek proba degerini aliyorum
for i in range(len(proba)):
    
    max_prob = np.max(proba[i,:])
    if max_prob > 0.95:
        idx.append(i)
        number_list.append(np.argmax(proba[i]))

#gorsellestirme islemlerini hallediyorum
for i in range(len(number_list)):
    
    j = idx[i]
    cv2.rectangle(image, (boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), (0,0,255), 2)
    cv2.putText(image, str(np.argmax(proba[j])),  (boxes[j, 0] + 5, boxes[j, 1] + 5),
                cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,255,0), 1)
    
    cv2.imshow("Image", image)

k = cv2.waitKey(0) & 0xFF #esc tusu

if k == 27: 
    cv2.destroyAllWindows()