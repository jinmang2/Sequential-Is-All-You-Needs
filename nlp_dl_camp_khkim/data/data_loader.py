# -*- encoding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
from glob import glob

# encoding 문제로 보류
def get_nsmc_data(return_pandas=True):
    paths = os.path.abspath('nlp_dl_camp_khkim/data/nsmc/raw/*.json')
    paths = [path.replace('\\', '/') for path in glob(paths)]
    res = []
    for path in paths:
        with open(path, encoding='utf-8') as data_file:
            res.extend(json.load(data_file))
    if return_pandas:
        df = pd.DataFrame(res)
        return df
    else:
        return res
