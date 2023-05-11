import pandas as pd
import numpy as np

data = pd.read_csv ("data_C02_emission.csv")

# a)
print(len(data)) #broj podataka odnosno redova
print(data.info()) #tip svake velicine
data.dropna(axis=0) #brisi prazna polja
data.drop_duplicates() #brisi duplikate

#pretvaranje stupaca u svakom retku u type category
for col in ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']:
    data[col] = data[col].astype('category')


#b)
sorted = data.sort_values(by=['Fuel Consumption City (L/100km)']) #sortiranje silazno

print("Najveca potrosnja: \n", sorted[["Make", "Model", "Fuel Consumption City (L/100km)"]].tail(3)) #ispis zadnja 3
print("Najmanja potrosnja: \n", sorted[["Make", "Model", "Fuel Consumption City (L/100km)"]].head(3)) #ispis prva tri

#c)
izdvojena_po_velicini_motora = data[(data["Engine Size (L)"] >= 2.5 ) & ( data ["Engine Size (L)" ] <= 3.5 )] #vozila koji zadovoljavaju uvjete

print (len(izdvojena_po_velicini_motora)) #broj vozila zadovoljenih uvjeta
print(izdvojena_po_velicini_motora["CO2 Emissions (g/km)"].mean()) #srednja vrijednos potrosnje goriva tih vozila
print("\n")

#d)
audi_cars = data[data["Make"] == "Audi"] #samo audi vozila
print("broj audi vozila: ", len(audi_cars)) #broj audi vozila
print(audi_cars[(data['Cylinders'] == 4)]['CO2 Emissions (g/km)'].mean()) #prosjecna co2 emisija za audi auta koji imaju 4 cilindra

#e)
data_g_by_cylinders = data.groupby('Cylinders') #grupiraj po cilindrima
print(data_g_by_cylinders['Cylinders'].count()) #koliko ima ata za svaki broj cilindara
print(data_g_by_cylinders['CO2 Emissions (g/km)'].mean()) #prosjecna co2 emisija za svaki broj cilindara

#f)
print(data[(data['Fuel Type'] == 'Z')]['Fuel Consumption City (L/100km)'].mean()) #prosjecna vrijednost potrosnje u gradu za benzince
print(data[(data['Fuel Type'] == 'Z')]['Fuel Consumption City (L/100km)'].median()) #medijan potrosnje u gradu za benzince

print(data[(data['Fuel Type'] == 'X')]['Fuel Consumption City (L/100km)'].mean()) #prosjecna vrijednost potrosnje u gradu za dizele
print(data[(data['Fuel Type'] == 'X')]['Fuel Consumption City (L/100km)'].median()) #medijan potrosnje u gradu za dizele

#g)
vozilo = data[(data["Cylinders"] == 4 ) & ( data['Fuel Type'] == 'X')] #pronadji auta koji zadovoljavaju uvjete
sortirana_vozila = vozilo.sort_values(by=['Fuel Consumption City (L/100km)']) #sortiraj te aute po potrosnji u gradu
print(sortirana_vozila.head(1)) #uzmi prvi auto

#h) 
print("Broj auta s rucnim mjenjacem: ", len(data[data["Transmission"].str.startswith("M")]))#broj vozila koji imaju rucni mjenjac

#i)
print(data.corr(numeric_only = True))
