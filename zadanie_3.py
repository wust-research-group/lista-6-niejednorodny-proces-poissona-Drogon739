import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def generate_poisson_process(intensity_function, T: float, dt: float = 0.01) -> np.ndarray:
    """
    Generuje zdarzenia dla niejednorodnego procesu Poissona.

    Parameters:
        intensity_function (callable): Funkcja intensywności w zależności od czasu.
        T (float): Czas końcowy symulacji.
        dt (float): Krok czasowy.

    Returns:
        np.ndarray: Tablica czasów przyjścia zdarzeń.
    """
    time_points = np.arange(0, T, dt)
    increments = np.random.poisson(intensity_function(time_points) * dt)
    events = time_points[increments > 0]
    return events

def intensity_function_1(t: float) -> float:
    """
    Funkcja intensywności dla procesu 1.

    Parameters:
        t (float): Czas.

    Returns:
        float: Wartość funkcji intensywności w punkcie czasowym t.
    """
    return 1 + 0.5 * t

def intensity_function_2(t: float) -> float:
    """
    Funkcja intensywności dla procesu 2.

    Parameters:
        t (float): Czas.

    Returns:
        float: Wartość funkcji intensywności w punkcie czasowym t.
    """
    return 4 - 0.2 * t

def cumulative_f_1(t: float) -> float:
    """
    Skumulowana funkcja intensywności dla procesu 1.

    Parameters:
        t (float): Czas.

    Returns:
        float: Wartość skumulowanej funkcji intensywności w punkcie czasowym t.
    """
    return t + 0.25 * t**2

def cumulative_f_2(t: float) -> float:
    """
    Skumulowana funkcja intensywności dla procesu 2.

    Parameters:
        t (float): Czas.

    Returns:
        float: Wartość skumulowanej funkcji intensywności w punkcie czasowym t.
    """
    return 4*t - 0.1 * t**2

def cumulative_events(events: np.ndarray, T: float, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oblicza skumulowaną liczbę zdarzeń w czasie.

    Parameters:
        events (np.ndarray): Tablica czasów przyjścia zdarzeń.
        T (float): Czas końcowy symulacji.
        dt (float): Krok czasowy.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tablice czasów i skumulowanej liczby zdarzeń.
    """
    time_points = np.arange(0, T, dt)
    event_counts = np.array([np.sum(events <= t) for t in time_points])
    return time_points, event_counts

# Parametry symulacji
T = 10  # Czas trwania procesu

# Generowanie zdarzeń dla dwóch procesów Poissona
process_1 = generate_poisson_process(intensity_function_1, T)
process_2 = generate_poisson_process(intensity_function_2, T)

# Łączenie zdarzeń z obu procesów
combined_process = np.sort(np.concatenate((process_1, process_2)))

# Obliczanie trajektorii procesów
time_line_1, cum_events_1 = cumulative_events(process_1, T)
time_line_2, cum_events_2 = cumulative_events(process_2, T)
time_line_combined, cum_events_combined = cumulative_events(combined_process, T)

# Teoretyczne skumulowane intensywności
theoretical_cumulative_intensity_1 = cumulative_f_1(time_line_1)
theoretical_cumulative_intensity_2 = cumulative_f_2(time_line_2)
theoretical_cumulative_intensity_combined = cumulative_f_1(time_line_combined) + cumulative_f_2(time_line_combined)

# Wykresy
plt.figure(figsize=(18, 10))

# Trajektorie procesów i funkcje intensywności
plt.plot(time_line_1, cum_events_1, label='Process 1')
plt.plot(time_line_2, cum_events_2, label='Process 2')
plt.plot(time_line_combined, cum_events_combined, label='Combined Process', linestyle='--')
plt.plot(time_line_1, theoretical_cumulative_intensity_1, label='F1(t)', linestyle='-.', alpha=0.7)
plt.plot(time_line_2, theoretical_cumulative_intensity_2, label='F2(t)', linestyle=':', alpha=0.7)
plt.plot(time_line_combined, theoretical_cumulative_intensity_combined, label='F1(t) + F2(t)', alpha=0.5)

# Dostosowanie osi i legendy
plt.xlabel('Time')
plt.ylabel('Cumulative Events / Intensity')
plt.title('Cumulative Event Trajectories and Intensity Functions')
plt.legend()
plt.grid(True)

# Wyświetlenie wykresu
plt.tight_layout()
plt.show()
