import importlib
import sys
import time

for i in range(5):  # Numero di esecuzioni con costanti diverse

    time.sleep(1)

    with open("config.py", "w") as f:
        f.write(f"M = {0.5 * i}\n")
        f.write(f"m = {0.2 * i}\n")
        f.write(f"b = {0.1 * i}\n")
        f.write(f"ftheta = {0.1 * i}\n")
        f.write(f"l = {0.3 * i}\n")

    time.sleep(1)

    if i == 0:
        import pendulum_model
    else:
        importlib.reload(pendulum_model)  # Ricarica pendulum_model con i nuovi valori
