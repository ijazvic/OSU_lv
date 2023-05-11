from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

#ucitaj ugradeni podatkovni skup
data = pd.read_csv ("data_C02_emission.csv")

#a)

#odabiranje ulaznih varijabli
input = ['Fuel Consumption City (L/100km)',
                   'Fuel Consumption Hwy (L/100km)',
                   'Fuel Consumption Comb (L/100km)',
                   'Fuel Consumption Comb (mpg)',
                   'Cylinders'
]

#odabiranje izlazne varijable
output = ['CO2 Emissions (g/km)']

#pretvaranje u numpy objekt
x = data[input].to_numpy()
y = data[output].to_numpy()

# podijeli skup na podatkovni skup za ucenje i podatkovni skup za testiranje
X_train , X_test , y_train , y_test = train_test_split (x , y , test_size = 0.2 , random_state =1 )


#b)

input_train = X_train[:,0] #odabirenje jedne ulazne vrijednosti iz skupa za ucenje
input_test = X_test[:,0] #odabirenje jedne ulazne vrijednosti iz skupa za testiranje

plt.scatter(input_train, y_train, c="#0000ff") #ovisnost jedne ulazne varijable o emisiji co2 TRENING
plt.scatter(input_test, y_test, c="#FF0000") #ovisnost jedne ulazne varijable o emisiji co2 TEST
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emission (g/km)')
plt.show()

#c)

# min - max skaliranje
sc = MinMaxScaler ()
X_train_n = sc.fit_transform ( X_train ) #skaliranje skupa za ucenje
X_test_n = sc.transform ( X_test ) #skaliranje skupa za testiranje

plt.hist(X_train[:,0], bins=20) #ispis histograma prije skaliranja
plt.title("potrosnja prije skaliranja")
plt.show()

plt.hist(X_train_n[:,0], bins=20) #ispis histograma poslije skaliranja
plt.title("potrosnja poslije skaliranja")
plt.show()


#d)

# inicijalizacija i ucenje modela logisticke regresije
linearModel = lm.LinearRegression()
linearModel.fit( X_train_n , y_train ) #dobili smo parametre funkcije, naucili smo racunalo kako da dodje do rezultata

print(linearModel.coef_) #parametri linearne regresije

#e) 

#procjena vrijednosti izlaza na osnvu ulaznih testnih podataka // predikcija na skupu podataka za testiranje
y_test_prediction = linearModel.predict(X_test_n) #dobivanje naseg izlaza na osnovi ucenja tj testnih podataka
plt.scatter(y_test, y_test_prediction, s=5) #usporedjivanje stvarnih emisija s dobivenim (izracunatim)
plt.xlabel('stvarne vrijednosti emisije')
plt.ylabel('dobivene vrijednosti emisije')
plt.show()


#f)

#izracun vrijednosti regresijskih metrika na skupu za testiranje
print("MAE: ", mean_absolute_error(y_test, y_test_prediction))
print("MSE: ", mean_squared_error(y_test, y_test_prediction))
print("RMSE: ", (mean_squared_error(y_test, y_test_prediction)) ** (1/2))
print("MAPE: ", mean_absolute_percentage_error(y_test, y_test_prediction))
print("R2: ", r2_score(y_test, y_test_prediction))


#g)

linearModel = lm.LinearRegression()
linearModel.fit(X_train_n[:, 3:], y_train)
y_test_prediction = linearModel.predict(X_test_n[:, 3:]) #uzmi sve vrijednosti ali izostavi prva 3 ulazna parametra

print('Metrike nakon izostavljanja prva tri ulazna parametra:')
print("MAE: ", mean_absolute_error(y_test, y_test_prediction))
print("MSE: ", mean_squared_error(y_test, y_test_prediction))
print("RMSE: ", (mean_squared_error(y_test, y_test_prediction)) ** (1/2))
print("MAPE: ", mean_absolute_percentage_error(y_test, y_test_prediction))
print("R2: ", r2_score(y_test, y_test_prediction))

#novi model je losije izracunao podatke jer je manje ulaznih velicina