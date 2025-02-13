import os
import importlib
import time
import numpy as np
from numpy.array_api import equal

from GLIS_BO_main import main
import sys
import pickle

perturb_pct = 0.2

i = 8

while i < 1000:  # Numero di esecuzioni con costanti diverse

    try: # try one calibration
        experiment_name = f"{i:04}"

        params = {}
        params['M'] = 0.5 * (1 + perturb_pct * np.random.uniform(low=-1.0, high=1.0))
        params['m'] = 0.2 * (1 + perturb_pct * np.random.uniform(low=-1.0, high=1.0))
        params['b'] = 0.1 * (1 + perturb_pct * np.random.uniform(low=-1.0, high=1.0))
        params['ftheta'] = 0.1 * (1 + perturb_pct * np.random.uniform(low=-1.0, high=1.0))
        params['l'] = 0.3 * (1 + perturb_pct * np.random.uniform(low=-1.0, high=1.0))

        # Run the code
        main(experiment_name, params)

        with open(os.path.join('results/params', experiment_name + '.pkl'), 'wb') as f:
            pickle.dump(params, f)
        i += 1 # next iter

    except Exception as e:
        print(e)
        continue # Go the the next one





