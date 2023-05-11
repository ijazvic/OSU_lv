import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

# 1.dio zadatka broj boja slike
print(len(np.unique(img_array_aprox,axis=0)))


# 2.dio zadatka
km = KMeans(n_clusters=25, init='random', n_init=5, random_state=0)
# pokretanje grupiranja primjera
km.fit(img_array_aprox)
# dodijeljivanje grupe svakom primjeru
labels = km.predict(img_array_aprox)
# ispis grupiranih podataka


# 3.dio zadatka
nova = km.cluster_centers_[labels]
nova = np.reshape(nova, (img.shape))


# 4.dio zadatka
flag, usporedba = plt.subplots(2, 1)
usporedba[0].imshow(img)
usporedba[1].imshow(nova)
plt.show()


# 5.dio zataka
for i in range (2, 7):
    img = Image.imread(f"imgs\\test_{i}.jpg")        
    img = img.astype(np.float64) / 255
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))
    img_array_aprox = img_array.copy()
    km = KMeans(n_clusters=6, init='random', n_init=5, random_state=0)
    km.fit(img_array_aprox)
    labels = km.predict(img_array_aprox)
    nova = km.cluster_centers_[labels]
    nova = np.reshape(nova, (img.shape))
    flag, usporedba = plt.subplots(2, 1)
    usporedba[0].imshow(img)
    usporedba[1].imshow(nova)
    plt.show()  

# 6.dio zadatka

K_ = range(1, 10)
J_ = []
for i in K_:
    km = KMeans(n_clusters=i, init='random', n_init=5, random_state=0)
    km.fit(img_array_aprox)
    J_.append(km.inertia_)
plt.plot(K_, J_) #prikaz vrijednost kriterijske funkcije J_ 
plt.show()


# 7.dio vjezbe

km = KMeans(n_clusters=4, init='random', n_init=5, random_state=0)
km.fit(img_array_aprox)
labels = km.predict(img_array_aprox)

unique_labels = np.unique(labels)
print(unique_labels)

flag, usporedba = plt.subplots(2, 2)

for i in range(len(unique_labels)):
    bit_values = labels==[i]
    bit_img = np.reshape(bit_values, (img.shape[0:2]))
    bit_img = bit_img*1
    x=int(i/2)
    y=i%2
    usporedba[x, y].imshow(bit_img)

plt.show()
