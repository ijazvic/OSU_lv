"""Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i
ženama. Skripta zadatak_2.py ucitava dane podatke u obliku numpy polja ˇ data pri cemu je u ˇ
prvom stupcu polja oznaka spola (1 muško, 0 žensko), drugi stupac polja je visina u cm, a treci´
stupac polja je masa u kg.

a) Na temelju velicine numpy polja data, na koliko osoba su izvršena mjerenja? ˇ
b) Prikažite odnos visine i mase osobe pomocu naredbe ´ matplotlib.pyplot.scatter.
c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.
d) Izracunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom ˇ
podatkovnom skupu.
e) Ponovite zadatak pod d), ali samo za muškarce, odnosno žene. Npr. kako biste izdvojili
muškarce, stvorite polje koje zadrži bool vrijednosti i njega koristite kao indeks retka.
ind = (data[:,0] == 1)"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.csv", skiprows = 1, delimiter = ",") #ucitavanje datoteke u obliku numpy polja
velicina = len(data) #velicina numpy polja
print(velicina)

#prolazi kroz sve retke, a pri tome uzima index [1] i indeks [2]
plt.scatter(data[:,1], data[:,2]) 
plt.ylabel("tezina")
plt.xlabel("visina")
plt.title("Omjer tezine i visine")
plt.show ()

#prolazi kroz svaki 50 redak, a pri tome uzima index [1] i indeks [2]
x = data[::50]
y = data[::50]
plt.scatter(x[:,1], y[:,2]) 
plt.ylabel("tezina")
plt.xlabel("visina")
plt.title("Omjer tezine i visine")
plt.show ()

#minimalna, maksimalna, srednja vrijednost podatkovnog skupa
print(np.min(data[:,1]), "minimalna visina")
print(np.max(data[:,1]), "maksimalna visina")
print(np.mean(data[:,1]), "srednja vrijednost visine\n")

#minimalna, maksimalna, srednja vrijednost podatkovnog skupa za muskarce
muskarci = (data[:,0] == 1) #gleda se prvi stupac i ako je true napuni polje muskarci s svim vrijednostima
print(np.min(data[muskarci,1]), "minimalna visina") #racunaj min visinu samo za muskarce
print(np.max(data[muskarci,1]), "maksimalna visina")
print(np.mean(data[muskarci,1]), "srednja vrijednost visine\n")

#minimalna, maksimalna, srednja vrijednost podatkovnog skupa za zene
zene = (data[:,0] == 0)
print(np.min(data[zene,1]), "minimalna visina")
print(np.max(data[zene,1]), "maksimalna visina")
print(np.mean(data[zene,1]), "srednja vrijednost visine")