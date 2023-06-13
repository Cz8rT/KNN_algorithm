import numpy as np
from collections import Counter


def euklides_dystans(x1, x2):  # Obliczenie odległości
    return np.sqrt(np.sum((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2 + (x1[2]-x2[2])**2 + (x1[3]-x2[3])**2))


class KNN:

    def __init__(self, k=3):
        self.k = k  # ilość rozpatrywanych sąsiadów

    def pobierz_dane(self, X, Y):
        self.X_treningowe = X  # atrybuty próbek treningowych
        self.Y_treningowe = Y  # klasy próbek treningowych

    def oszacuj(self, probka):  # Oszacowanie klasy pojedynczej próbki testowej
        # Obliczanie odległości do sąsiadów
        odleglosci = [euklides_dystans(probka, element)
                      for element in self.X_treningowe]
        # Indeksy najbliższych k-sąsiadów
        k_sasiedzi_indeksy = np.argsort(odleglosci)[:self.k]
        # Klasy najbliższych k-sąsiadów
        k_sasiedzi_klasy = [self.Y_treningowe[i][0]
                            for i in k_sasiedzi_indeksy]
        # Głosowanie większościowe
        # Pierwszy index wskazuje na najczęstszą klasę wśród sąsiadów
        most_common = Counter(k_sasiedzi_klasy).most_common(1)
        # Zwrócenie tablicy z najczęstszą klasą
        return [most_common[0][0]]
