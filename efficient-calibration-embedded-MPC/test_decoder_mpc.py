import os
import importlib
import time
import numpy as np

from GLIS_BO_decoder_main import main
import sys
import pickle

# Load the parameters of the test set
samples = range(1251,1300)

for i in samples:

    print(f'Optimizing experiment {i}')
    experiment = i
    experiment_name = f"{i:04}"

    with open(f'results/params/{experiment:04}.pkl', 'rb') as f:
        params = pickle.load(f)

    try: # try one calibration
        main(experiment_name, params)
        print(f'Finished optimization of experiment {i}')
    except Exception as e:
        print(e)






