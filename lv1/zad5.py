"""Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva ˇ SMSSpamCollection.txt
[1]. Ova datoteka sadrži 5574 SMS poruka pri cemu su neke ozna ˇ cene kao ˇ spam, a neke kao ham.
Primjer dijela datoteke:"""

file = open("SMSSpamCollection.txt")
hamMsg = []
spamMsg= []
hamwordCounter = 0
spamwordCounter = 0
usklicnik = 0

for line in file:
    line = line.strip().split()
    if line[0] == "ham":
        hamMsg.append(line[1:])
        hamwordCounter += len(line[1:])
    else:
        spamMsg.append(line[1])
        spamwordCounter += len(line[1:])
        if line[-1].endswith("!"):
            usklicnik += 1

prosjekham = float(hamwordCounter)/len(hamMsg)
prosjekspama = float(spamwordCounter)/len(spamMsg)

print("Prosjecan broj rijeci u ham poruci: ", prosjekham)
print("Prosjecan broj rijeci u spam poruci: ", prosjekspama)
print("broj spam poruka koje završavaju uskličnikom: ", usklicnik)