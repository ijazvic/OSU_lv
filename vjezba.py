#import bibilioteka
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#učitavanje dataseta
iris = datasets.load_iris()
data = datasets.load_iris()

df = pd.DataFrame(data=data.data, columns=data.feature_names)

data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])


##################################################
#1. zadatak
"""Iris Dataset sastoji se od informacija o laticama i cašicama tri razli ˇ cita cvijeta ˇ
irisa (Setosa, Versicolour i Virginica). Dostupan je u sklopu bibilioteke scikitlearn:

from sklearn import datasets
iris = datasets.load_iris()

Upoznajte se s datasetom i dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na ´
sljedeca pitanja: ´
a) Prikažite odnos duljine latice i cašice svih pripadnika klase Versicolour pomo ˇ cu´ scatter
dijagrama. Dodajte na isti dijagram odnos duljine latice i cašice svih pripadnika klase ˇ
Virginica, drugom bojom. Dodajte naziv dijagrama i nazive osi te legendu. Komentirajte
prikazani dijagram.
b) Pomocu stup ´ castog dijagrama prikažite prosje ˇ cnu vrijednost širine ˇ cašice za sve tri klase ˇ
cvijeta. Dodajte naziv dijagrama i nazive osi. Komentirajte prikazani dijagram.
c) Koliko jedinki pripadnika klase Virginica ima vecu širinu ´ cašice od prosje ˇ cne širine ˇ cašice ˇ
te klase?"""
##################################################


#a)
versicolour_data = (data1[ data1['target'] == 1])
plt.scatter(versicolour_data[['sepal length (cm)']], versicolour_data[['petal length (cm)']], c = "red", label="versicolour")
plt.title('odnos duljine latice i cašice svih pripadnika klase versicolour')
plt.show()
virginica_data = (data1[ data1['target'] == 2])
plt.scatter(versicolour_data[['sepal length (cm)']], versicolour_data[['petal length (cm)']], c = "red", label="versicolour")
plt.scatter(virginica_data[['sepal length (cm)']], virginica_data[['petal length (cm)']], c = "blue", label="virginica")
plt.title('odnos duljine latice i cašice svih pripadnika klase versicolour i virginica')
plt.xlabel("sepal length (cm)")
plt.ylabel("petal length (cm)")
plt.legend(loc="upper left")
plt.show()
#b)
grouped = data1.groupby('target').mean()
grouped["sepal width (cm)"].plot(kind="bar")
plt.title("prosječna vrijednost širine cašice za sve tri klase")
plt.xlabel("klase")
plt.ylabel("prosječna širina čašice")
plt.show()
#c)
virginica_avg = virginica_data["sepal width (cm)"].mean()
count = virginica_data[virginica_data["sepal width (cm)"] > virginica_avg].count()
print(count)


##################################################
#2. zadatak
"""Iris Dataset sastoji se od informacija o laticama i cašicama tri razli ˇ cita cvijeta ˇ
irisa (Setosa, Versicolour i Virginica). Dostupan je u sklopu bibilioteke scikitlearn:

from sklearn import datasets
iris = datasets.load_iris()

Upoznajte se s datasetom. Pripremite podatke za ucenje. Dodajte programski kod u skriptu ˇ
pomocu kojeg možete odgovoriti na sljede ´ ca pitanja: ´
a) Pronadite optimalni broj klastera ¯ K za klasifikaciju cvijeta irisa algoritmom K srednjih
vrijednosti.
b) Graficki prikažite lakat metodu. ˇ
c) Primijenite algoritam K srednjih vrijednosti koji ce prona ´ ci grupe u podatcima. Koristite ´
vrijednot K dobivenu u prethodnom zadatku.
d) Dijagramom raspršenja prikažite dobivene klastere. Obojite ih razlicitim bojama. Centroide ˇ
obojite crvenom bojom. Dodajte nazive osi, naziv dijagrama i legendu. Komentirajte
prikazani dijagram.
e) Usporedite dobivene klase sa njihovim stvarnim vrijednostima. Izracunajte to ˇ cnost klasi ˇ fikacije"""
##################################################

#a) i b)
input_variables = ["sepal width (cm)", 'petal width (cm)', 'sepal length (cm)', 'petal length (cm)']

#b)
X = data1[input_variables].to_numpy()
Js = []
Ks = range(1, 8)
for i in range(1, 8):
    km = KMeans(n_clusters=i, init='random', n_init=5, random_state=0)

    km.fit(X)
    Js.append(km.inertia_)
    labels = km.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=km.labels_.astype(float))
    plt.title(f'K={i}')
    plt.show()
plt.plot(Ks, Js)
plt.show()

#c)
km = KMeans(n_clusters=3, init='random', n_init=5, random_state=0)
km.fit(X)
labels = km.predict(X)

#d)
plt.scatter(X[:, 0], X[:, 1], c=km.labels_.astype(float))
centroids = km.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
plt.scatter(centroids_x,centroids_y,marker = "x", s=50,linewidths = 5, zorder = 10, c='red')
plt.title("Dijagramom raspršenja ")
plt.xlabel("sepal width (cm)")
plt.ylabel("petal width (cm)")
plt.legend(['Setosa','centroidi', 'Virginica'], loc='upper left')
plt.show()
#e)
score = metrics.accuracy_score(data1['target'],labels)
print("Tocnost klasifikacije je: ", score)


##################################################
#3. zadatak
"""Iris Dataset sastoji se od informacija o laticama i cašicama tri razli ˇ cita cvijeta ˇ
irisa (Setosa, Versicolour i Virginica). Dostupan je u sklopu bibilioteke scikitlearn:

from sklearn import datasets
iris = datasets.load_iris()

Upoznajte se s datasetom. Podijelite ga na ulazne podatke X i izlazne podatke y predstavljene
klasom cvijeta. Pripremite podatke za ucenje neuronske mreže (kategori ˇ cke veli ˇ cine, skaliranje...). ˇ
Podijelite podatke na skup za ucenje i skup za testiranje modela u omjeru 75:25. Dodajte ˇ
programski kod u skriptu pomocu kojeg možete odgovoriti na sljede ´ ca pitanja: ´
a) Izgradite neuronsku mrežu sa sljedecim karakteristikama: ´
- model ocekuje ulazne podatke ˇ X
- prvi skriveni sloj ima 10 neurona i koristi relu aktivacijsku funkciju
- drugi skriveni sloj ima 7 neurona i koristi relu aktivacijsku funkciju
- treci skriveni sloj ima 5 neurona i koristi ´ relu aktivacijsku funkciju
- izlazni sloj ima 3 neurona i koristi softmax aktivacijsku funkciju.
-izmedu prvog i drugog te drugog i tre ¯ ceg sloja dodajte ´ Dropout sloj s 30% izbacenih ˇ
neurona
Ispišite informacije o mreži u terminal.
b) Podesite proces treniranja mreže sa sljedecim parametrima: ´
- loss argument: categorical_crossentropy
- optimizer: adam
- metrika: accuracy.
c) Pokrenite ucenje mreže sa proizvoljnim brojem epoha (pokušajte sa 500) i veli ˇ cinom ˇ
batch-a 7.
d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju ucitanog modela. ˇ
e) Izvršite evaluaciju mreže na testnom skupu podataka.
f) Izvršite predikciju mreže na skupu podataka za testiranje. Prikažite matricu zabune za skup
podataka za testiranje. Komentirajte dobivene rezultate i predložite kako biste ih poboljšali,
ako je potrebno."""
##################################################


#predobrada podataka
input_variables = ["sepal width (cm)", 'petal width (cm)', 'sepal length (cm)', 'petal length (cm)']
output_variables = ['target']
X = data1[input_variables].to_numpy()
y = data1[output_variables].to_numpy()

(X_train, X_test, y_train, y_test) = train_test_split(
    X, y, test_size=0.25, random_state=1)

sc = StandardScaler()
X_train_n = pd.DataFrame(sc.fit_transform(X_train))
X_test_n = pd.DataFrame(sc.transform(X_test))
#a)
model = keras.Sequential()
model.add(layers.Input(shape=(4,)))
model.add(layers.Dense(10, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(7, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(5, activation="relu"))
model.add(layers.Dense(3, activation="softmax"))
model.summary()
#b)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy",])
#c)
history = model.fit(X_train, y_train, batch_size=7, epochs=500, validation_split=0.1)
#d)
keras.models.save_model(model, "model")
#e)
model = keras.models.load_model("model")
score = model.evaluate(X_test, y_test, verbose=0)
#f)
y_test_pred = model.predict(X_test)             
y_test_pred = np.argmax(y_test_pred, axis=1)   

cm = confusion_matrix(y_test, y_test_pred)
print("Matrica zabune:", cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred))
disp.plot()
plt.show()