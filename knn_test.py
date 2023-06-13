from scripts.knn import KNN
import numpy as np

# pobranie danych z pliku '.txt'
plik = open('./iris.txt', "r")
dane = plik.read().split("\n")
plik.close()

# Tablica przechowująca poszczególne linie danych
lista_danych = []

# Konwersja i przeformatowanie poszczególnych linii danych
for wiersz in dane:
    linijka = wiersz.split('\t')
    lista_float = []
    for element in linijka:
        lista_float.append(float(element))
    lista_danych.append(lista_float)

lista_danych = np.array(lista_danych)

# Posortowanie tablicy względem kolumny klas
lista_danych = lista_danych[lista_danych[:, 4].argsort(), :]


# Funkcja porównująca każdą z próbek i obliczająca średnią poprawność algorytmu dla podanego 'k'
def Walidacja(array, k=5):
    print("\nAlgorytm KNN")
    print("Współczynnik k:", k, "\n")
    # Licznik poprawnie oszacowanych próbek przez KNN
    licznik_poprawnych = 0
    for i in range(len(array)):
        probka_testowa = array[i]
        probki_treningowe = np.delete(array, i, 0)

        # Podział próbek treningowych na wymiary i klasy
        wymiary_treningowe = probki_treningowe[:, 0:4]
        klasy_treningowe = probki_treningowe[:, 4:]

        # Podział próbki testowej na wymiary i klasę
        wymiary_testowe = probka_testowa[0:4]
        klasa_testowa = probka_testowa[4:]

        # Wywołania algorytmu KNN dla podstawionych danych
        klasyfikacja = KNN(k)
        klasyfikacja.pobierz_dane(wymiary_treningowe, klasy_treningowe)
        predykcja = klasyfikacja.oszacuj(wymiary_testowe)
        if (predykcja == klasa_testowa):
            licznik_poprawnych += 1
        else:
            print("Błędnie oszacowana próbka:", probka_testowa,
                  "; klasa oszacowana przez algorytm KNN:", predykcja)
    # Obliczenie dokładności
    print("\nPoprawnie oszacowanych próbek:", licznik_poprawnych)
    dokladnosc = licznik_poprawnych / len(array) * 100
    print("Obliczona dokładność klasyfikacji testowych próbek:",
          round(dokladnosc, 2), "%\n")


Walidacja(lista_danych, 5)
