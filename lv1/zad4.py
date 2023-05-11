"""Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva ˇ song.txt.
Potrebno je napraviti rjecnik koji kao klju ˇ ceve koristi sve razli ˇ cite rije ˇ ci koje se pojavljuju u ˇ
datoteci, dok su vrijednosti jednake broju puta koliko se svaka rijec (klju ˇ c) pojavljuje u datoteci. ˇ
Koliko je rijeci koje se pojavljuju samo jednom u datoteci? Ispišite ih."""

rijecnik = {}
file = open("song.txt")

for line in file:
    for word in line.split():
        if word in rijecnik.keys():
            rijecnik[word] += 1
        else:
            rijecnik[word] = 1

for word in rijecnik.keys():
    if rijecnik[word] == 1:
        print(word)

file.close()