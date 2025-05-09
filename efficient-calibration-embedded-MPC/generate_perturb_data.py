import os
import importlib
import time
import numpy as np
from numpy.array_api import equal

from GLIS_BO_main import main
import sys
import pickle

perturb_pct = 0.2

i = 1000

while i < 2000:  # Numero di esecuzioni con costanti diverse

    try: # try one calibration
        experiment_name = f"{i:04}"

        lb = np.array([0.5, 0.2, 0.1, 0.1, 0.3])
        ub = lb * 5

        keys = ["M", "m", "b", "ftheta", "l"]
        # Generate random values uniformly between lb and ub
        random_vals = np.random.uniform(lb, ub)

        # Create the dictionary
        params = dict(zip(keys, random_vals))

        # Run the code
        main(experiment_name, params)

        with open(os.path.join('results/params', experiment_name + '.pkl'), 'wb') as f:
            pickle.dump(params, f)

        i += 1 # next iter

    except Exception as e:
        print(e)
        continue # Go the the next one





