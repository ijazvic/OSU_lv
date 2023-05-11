"""Napišite program koji od korisnika zahtijeva unos brojeva u beskonacnoj petlji ˇ
sve dok korisnik ne upiše „Done“ (bez navodnika). Pri tome brojeve spremajte u listu. Nakon toga
potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu
vrijednost. Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
(npr. slovo umjesto brojke) na nacin da program zanemari taj unos i ispiše odgovaraju ˇ cu poruku."""

list = []
flag = 1

while flag == 1:
    unos = input("Unesi broj:")
    if(unos =="Done"):
        break
    else:
        try:
            broj = float(unos) #provjera da li je unesen broj ili slovo
            list.append(broj)
        except:
            print("nisi unjeo broj")

list.sort()
print(list)

print("brojeva u listi: ",len(list))
print("srednja vrijednost liste: ", sum(list) / len(list))
print("maks vrijednost u listi: ", max(list))
print("min vrijednost u listi: ", min(list))
