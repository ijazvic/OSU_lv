"""Skripta zadatak_3.py ucitava sliku ’ ˇ road.jpg’. Manipulacijom odgovarajuce´
numpy matrice pokušajte:
a) posvijetliti sliku,
b) prikazati samo drugu cetvrtinu slike po širini, ˇ
c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu,
d) zrcaliti sliku."""

import numpy as np
import matplotlib.pyplot as plt

#ucitaj sliu
img = plt.imread ("road.jpg")

#prikazi sliku
plt.figure()
plt.imshow(img)
plt.title("Normalna")
plt.show()


#prosvjetli sliku
svjetlija = img + 100 #prosvjetli za vrijednost 100
svjetlija[img>155] = 255 #ako neki piksel ima vecu vrijednosti od 155 postavi ga na 255 da ne bi bio crn

plt.figure()
plt.imshow(svjetlija)
plt.title("Svjetlija slika")
#moze se i samo s ovim prosvjetlit slika: plt.imshow(img, cmap="gray", alpha=0.5)
plt.show()


#prikaz druge cetvrtine slike
flag = img.shape[1]
druga_cetvrtina = img[:, int(flag/4): int(flag/2)]

plt.figure()
plt.imshow(druga_cetvrtina)
plt.title("Druga cetvrtina")
plt.show()


#rotiranje za 90 stupnjeva u smjeru kazalje na satu
img90 = np.rot90(img)
img90 = np.rot90(img90)
img90 = np.rot90(img90)

plt.figure()
plt.imshow(img90)
plt.title("Rotirana za 90 stupnjeva")
plt.show()

#zrcaljenje slike
plt.figure()
plt.imshow(np.fliplr(img))
plt.title("zrcalita slika")
plt.show()

