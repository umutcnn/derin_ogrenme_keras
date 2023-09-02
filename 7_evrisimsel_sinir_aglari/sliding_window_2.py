#kutuphaneleri import ediyoruz
import cv2
import matplotlib.pyplot as plt

#gorsellestirmek icin bu fonk kullanacagiz
#step dikdortgenin kac piksel atlayarak gezecegi
#ws gikdortgenin boyutunu belirler
def sliding_window(image, step, ws):
    for y in range(0, image.shape[0]-ws[1], step):
        for x in range(0, image.shape[1]-ws[0], step):
            
            #resim uzerindeki ilgili bolgeyi reurn ediyoruz
            yield(x,y, image[y:y+ws[1], x:x+ws[0]])
         
img = cv2.imread("husky.jpg")
im = sliding_window(img, 50, (200,150))

for i, image in enumerate(im):
    print(i)
    if i == 100:
        print(image[0], image[1])
        plt.imshow(image[2])
        
