from matplotlib import colors
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split


#generira umjetni binarni klasifikacijski problem s dvijeulazne velicine
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#a)

plt.scatter(X_train[:,0], X_train[:,1], marker="o", c=y_train, s=10, cmap=colors.ListedColormap(["red", "blue"])) #ulazni podaci za treniranje obojani crvenom, a izlazni podaci za treniranje plavom
plt.scatter(X_test[:,0], X_test[:,1], marker="x", c=y_test, s=20, cmap=colors.ListedColormap(["red", "blue"])) 
plt.show()

#b) izgradnja modela logisticke regresije na temelju skupa podataka za ucenje

#inicijalizacija i ucenje modela logisticke regresije
LogRegression_model = LogisticRegression()
LogRegression_model.fit( X_train , y_train )

#c) pronalazak atributa izgradjenog modela parametre modela i prikaz granice odluke naucenog modela

theta_0 = LogRegression_model.intercept_
theta_1 = LogRegression_model.coef_[0][0]
theta_2 = LogRegression_model.coef_[0][1]

a = -theta_1/theta_2
c = -theta_0/theta_2
xymin, xymax = -4, 4
xd = np.array([xymin, xymax])
yd = a*xd + c
plt.plot(xd, yd, linestyle='--')
plt.show()


#d) klasifikacija skupa podataka, matrica zabune na testnim podacima

# predikcija na skupu podataka za testiranje
y_test_p = LogRegression_model.predict ( X_test )

cm = confusion_matrix( y_test , y_test_p ) #matrica zabune
print ("Matrica zabune: " , cm )
disp = ConfusionMatrixDisplay ( confusion_matrix ( y_test , y_test_p ) ) #prikaz matrice zabune
disp.plot ()
plt.show ()

print (" Tocnost : " , accuracy_score( y_test, y_test_p ) ) #tocnost
print('Preciznost:', precision_score(y_test, y_test_p)) #preciznost
print('Odziv:', recall_score(y_test, y_test_p)) #odziv
print(classification_report(y_test, y_test_p)) #izvjestaj klasifikacije


#e) prikaz skupa za testiranje u ravnini

y_color = (y_test == y_test_p)
#zelena boja su dobro klasificirani primjeri, crvena pogresno klasificirani primjeri
plt.scatter(X_test[:, 0], X_test[:, 1], marker="o", c=y_color, s=25, cmap=colors.ListedColormap(["red", "green"])) 
plt.show()

