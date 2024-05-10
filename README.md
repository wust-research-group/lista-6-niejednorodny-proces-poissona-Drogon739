[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/lHr2qw8p)
# Symulacje Komputerowe

Wydział Matematyki, Semestr Letni 2023/2024

**Laboratorium 6:** Niejednorodny proces Poissona

**Termin oddania:** 16/05/2024 godz. 23:59:59 CET.

---

**Zadanie 1.** Metoda przerzedzania 
> Napisz algorytm, który dla ustalonej niejednorodnej funkcji intensywności $\lambda(t)$ generuje czasy oczekiwania niejednorodnego procesu Poissona metodą przerzedzania.
> - Sprawdź poprawność jego działania weryfikując, że $N_t \sim Poiss(\int_{0}^{t}\lambda{(s)}\mbox{ds})$

**Zadanie 2.** Metoda odwrotnej dystrybuanty
> Dla przypadków, w których możemy łatwo wyznaczać $m(t) =\int_{0}^{t}\lambda{(s)}\mbox{ds}$ można użyć alternatywnego algorytmu.
> 1. Dla ustalonego $T$ wygenerować zmienną $N_t \sim Poiss(m(T))$.
> 1. Za czasy oczekiwania wziąć $N_t$ posortowanych zmiennych o dystrybuancie $F(t) = m(t)/m(T)$.
> - Sprawdź również jego działanie.

**Zadanie 3.** Łączenie procesów Poissona
> Mając 2 niezależne niejednorodne procesy Poissona stwórz proces składający się z połączenia wszystkich zdarzeń obu.
> - Sprawdź, że jego intensywność jest sumą 2 oryginalnych intensywności.

**Zadanie 4.** Rozłączenie procesów Poissona
> Znacznikowanym procesem Poissona nazywamy proces Poissona, w którym każdy jego czas zdarzenia $S_i$ ma dołączony znacznik (etykietę) $Z_i$, które są niezależnymi zmiennymi losowymi.
> - Sprawdź, że dla znaczników $Z_i \in \{1, 2,\ldots, N\}$ zadanych rozkładem $P(Z_i = k) = p_k(S_i)$ znaniczkowane procesy otrzymane wybieraniem $S_i = k$ to niezależne niejednorodne procesy Poissona o intensywnościach $p_k(t)λ$.

**Uwaga**: Rozwiązania poszczególnych zadań należy umieścić w dedykowanych plikach `*.py` lub `*.ipynb` o nazwach `zadanie_1.[py|ipynb]`, `zadanie_2.[py|ipynb]` i `zadanie_3.[py|ipynb]`, itd.
