"""Napišite program koji od korisnika zahtijeva unos radnih sati te koliko je placen ´
po radnom satu. Koristite ugradenu Python metodu ¯ input(). Nakon toga izracunajte koliko ˇ
je korisnik zaradio i ispišite na ekran. Na kraju prepravite rješenje na nacin da ukupni iznos ˇ
izracunavate u zasebnoj funkciji naziva ˇ total_euro."""


radni_sat = float(input("unesi radne sate: "))
satnica = float(input("unesi satnicu: "))

zarada = radni_sat * satnica

print("zarada: ", zarada )
