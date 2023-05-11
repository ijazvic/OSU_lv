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
from sklearn.preprocessing import OneHotEncoder

#ucitaj ugradeni podatkovni skup
data = pd.read_csv ("data_C02_emission.csv")

#odabiranje ulaznih varijabli
input = [   "Make",
            'Fuel Consumption City (L/100km)',
            'Fuel Consumption Hwy (L/100km)',
            'Fuel Consumption Comb (L/100km)',
            'Fuel Consumption Comb (mpg)',
            'Cylinders',
            "Fuel Type"
]

#kodiranje kategoricke velicine 
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[["Fuel Type"]]).toarray()
data["Fuel Type"] = X_encoded

#odabiranje izlazne varijable
output = ['CO2 Emissions (g/km)']

#pretvaranje u numpy objekt
x = data[input].to_numpy()
y = data[output].to_numpy()

# podijeli skup na podatkovni skup za ucenje i podatkovni skup za testiranje
X_train , X_test , y_train , y_test = train_test_split (x , y , test_size = 0.2 , random_state =1 )

#izgradnja linearnog regresijskog modela
linearModel = lm.LinearRegression()
linearModel.fit( X_train[:,1:] , y_train ) #dobili smo parametre funkcije, naucili smo racunalo kako da dodje do rezultata
 
#procjena vrijednosti izlaza na osnvu ulaznih testnih podataka
y_test_prediction = linearModel.predict(X_test[:,1:]) #dobivanje naseg izlaza na osnovi ucenja tj testnih podataka, prvi je izostavljen jer je kategoricna vrijednost

#izracun vrijednosti regresijskih metrika na skupu za testiranje
print("MAE: ", mean_absolute_error(y_test, y_test_prediction))
print("MSE: ", mean_squared_error(y_test, y_test_prediction))
print("RMSE: ", (mean_squared_error(y_test, y_test_prediction)) ** (1/2))
print("MAPE: ", mean_absolute_percentage_error(y_test, y_test_prediction))
print("R2: ", r2_score(y_test, y_test_prediction))

max_pogreska = abs(y_test_prediction - y_test) #maksimalna pogreska u procjeni emsije co2 plinova
max_pogreska_index = np.argmax(max_pogreska) #trazenje indexa maksimalne pogreske
print("vrijednost maksimalne pogreske:", max_pogreska[max_pogreska_index])
print("Model s najvecom greskom: ", X_test[max_pogreska_index,0])
print("Predvidjeno:",y_test_prediction[max_pogreska_index], " stvarno:", y_test[max_pogreska_index])