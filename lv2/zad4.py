"""Napišite program koji ce kreirati sliku koja sadrži ´ cetiri kvadrata crne odnosno ˇ
bijele boje (vidi primjer slike 2.4 ispod). Za kreiranje ove funkcije koristite numpy funkcije
zeros i ones kako biste kreirali crna i bijela polja dimenzija 50x50 piksela. Kako biste ih složili
u odgovarajuci oblik koristite numpy funkcije ´ hstack i vstack."""

import numpy as np
import matplotlib.pyplot as plt

#pravljenje crnog i bijelog kvadratica
crna = np.zeros([50,50])
bijela = np.ones([50,50])

#spajanje prvog i drugog reda odnosno kvadratica
prvi_red = np.vstack((crna,bijela))
drugi_red = np.vstack((bijela,crna))

#horizontalno spajanje tj. jedno ispod drugog
img = np.hstack((prvi_red, drugi_red))

plt.figure()
plt.imshow(img, cmap="gray")
plt.title("crno bijela slika")
plt.show()
