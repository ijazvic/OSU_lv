"""Napišite program koji od korisnika zahtijeva upis jednog broja koji predstavlja nekakvu ocjenu i nalazi se izmedu 0.0 i 1.0. Ispišite kojoj kategoriji pripada ocjena na temelju
sljedecih uvjeta"""


print("Unesi broj od 0-1:")

try:
    number = float(input())

    if(number>1.0 or number<0.0):
        print("Uneseni broj nije unutar granica.")
    elif(number>=0.9 and number <=1.0):
        print("A")
    elif(number>=0.8 and number<0.9):
        print("B")
    elif(number>=0.7 and number<0.8):
        print("C")
    elif(number>=0.6 and number<0.7):
        print("D")
    elif(number>=0.0 and number<0.6):
        print("F")
except:
    print("You didn't enter a number.")
