#kutuphanelerimizi import ediyoruz

import keyboard #klavye icin
import uuid #ekrandan kayit almak icin 
import time #sure icin
from PIL import Image
from mss import mss

# referans alacagimiz noktalari aliyoruz. bu kare içerisindeki engellere gore yukari assagi saga sola yapacagiz
mon = {"top":300, "left":770, "width":250, "height":100}

#bu pikseller dogrultusunda ekrandaki o bolgeyi kesip frame olusturur.
sct = mss()

i = 0

def record_screen(record_id, key):
    global i
    
    i += 1
    # hangi tusa bastigimizi ve tuslara kac kez bastigimiz yazdiriyorum
    print("{}: {}".format(key, i))

    #yukaridaki belirledigim pikseller dogrultusunda ekrani al
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)

    #resmi kaydediyorum
    im.save("./img/{}_{}_{}.png".format(key, record_id, i))
    
#fonksitondan cikmati saglar
is_exit = False

def exit():
    global is_exit
    is_exit = True

#esc basinca exit fonksiyonunu cagiriyorum
keyboard.add_hotkey("esc", exit)

record_id = uuid.uuid4()

while True:
    
    #ciktin mi
    if is_exit: break

    try:
        #eger yukari tusuna bastiysan
        if keyboard.is_pressed(keyboard.KEY_UP):
            #ekrani al
            record_screen(record_id, "up")
            time.sleep(0.1)#0.1 saniye bekle

        #eger assagi tusuna bastiysan
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            #ekrani al
            record_screen(record_id, "down")
            time.sleep(0.1)#0.1 saniye bekle
            
        #eger hicbir tusa basmadiysan    
        elif keyboard.is_pressed("right"):
            record_screen(record_id, "right")
            time.sleep(0.1)#0.1 saniye bekle
    #hata almamizi engeller
    except RuntimeError: continue


