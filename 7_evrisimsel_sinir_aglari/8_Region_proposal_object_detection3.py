from keras.applications.resnet50 import preprocess_input
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2

from non_max_supression_3 import non_max_suppression

# seçmeli arama icin bir fonksiyon olusturduk
def selective_search(image):
    print("selective_search calisti")

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    #gorselimizi veriyoruz
    ss.setBaseImage(image)
    
    ss.switchToSelectiveSearchQuality()

    rects = ss.process()
    
    return rects[:1000]

# model
model = ResNet50(weights = "imagenet")
#resmi ice aktaralim
image = cv2.imread("animals.png")
image = cv2.resize(image, dsize=(400,400))
(H, W) = image.shape[:2]

# selective search calistiralim
rects = selective_search(image)

proposals = []
boxes = []

for (x, y, w, h) in rects:
    

    if w / float(W) < 0.1 or float(H) < 0.1: continue

    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224,224))
    
    roi = img_to_array(roi)
    roi = preprocess_input(roi)
    
    proposals.append(roi)
    boxes.append((x, y, w, h))
    
proposals = np.array(proposals)

print("tahmin yani predict islemimizi gerceklestiriyoruz")
preds = model.predict(proposals)
preds = imagenet_utils.decode_predictions(preds, top=1)

labels = {}
min_conf = 0.8

for (i, p) in enumerate(preds):

    # tahminler 3 tane değer dondurulecek 1. imageNet_id, 2. label(sınıf)
    # 3. yüzde tahmin değeri.
    (_, label, prob) = p[0]

    #eger prob, min_conf tan buyukse
    if prob >= min_conf:
        #ozzetle min_conf degerinden yuksek tahminleri isleme aliyoruz

        box = (x, y, x + w, y + h)
        #kutu ve olasilik
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L
        
        
clone = image.copy()

for label in labels.keys():

    #kutucuklari cizdiriyorum
    for (box, prob) in labels[label]:
        # non-maxima labelleri boluyoruz (kutular ve olasiliklar olarak)
        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])

        #kutucukların icersinden max olasiliği olani al
        boxes = non_max_suppression(boxes, proba)

        #sonra bunu ciz
        for (startX, startY, endX, endY) in boxes:
            cv2.rectangle(clone, (startX, startY), (endX, endY), (0,0,255), 2)
            y = startY - 10 if startY -10 > 10 else startY + 10
            cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)


        cv2.imshow("After", clone)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
        

k = cv2.waitKey(0) & 0xFF #esc tusu

if k == 27: 
    cv2.destroyAllWindows()


