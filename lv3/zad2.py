import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv ("data_C02_emission.csv")
data.dropna(axis=0)
data.drop_duplicates() 
data = data.reset_index(drop=True)

#a)
plt.figure()
data ["CO2 Emissions (g/km)"].plot( kind ="hist", bins = 20 ) #prikaz pomocu histograma emisiju co2 plinove
plt.show()

#b)
colordict = {'X': 'blue', 'Z':"red", 'D':"black", 'E':"green", 'N':"yellow"} #boje za razlicite tipove goriva
plt.scatter(data['Fuel Consumption City (L/100km)'], data["CO2 Emissions (g/km)"], c=[colordict[x] for x in data['Fuel Type']]) #scatter prikaz za svaki tip goriva
plt.show()

#c)
data.boxplot(column =["Fuel Consumption Hwy (L/100km)"], by="Fuel Type") #prikaz potrosnje goriva po tipu goriva
plt.show()

#d)
grupirano = data.groupby("Fuel Type") #grupiraj po gorivu
grupirano['Make'].count().plot(kind="bar") #prikaz broja auta po gorivu
plt.show()

#e)
grupirano2 = data.groupby("Cylinders") #grupiraj po cilindrima
grupirano2['CO2 Emissions (g/km)'].mean().plot(kind="bar") #prikaz prosjecne emisije po cilindru
plt.show()

