# -*- coding: utf-8 -*-

import sys
import os
import os.path as osp
import glob
import time
import numpy as np
from ddf_utils.factory.common import download
from multiprocessing import Pool
from functools import partial


source_dir = 'source/povcalnet'
all_brackets = np.logspace(-7, 13, 501, base=2, endpoint=True)
url_tmpl = "https://api.worldbank.org/pip/v1/pip?country=all&year=all&povline={}&fill_gaps=true&group_by=none&welfare_type=all&reporting_level=all&format=csv"
POOLSIZE = 4


def process(i, resume=True):
    bracket = all_brackets[i]
    if bracket > 2700:  # there is a limit in API
        bracket = 2700
    url = url_tmpl.format(bracket)
    file_csv = osp.join(source_dir, "{:04d}.csv".format(i))
    print(file_csv)
    download(url,
             file_csv,
             resume=resume,
             progress_bar=False,
             backoff=5,
             timeout=60)
    time.sleep(5)


if __name__ == "__main__":
    # create the folder
    os.makedirs(source_dir, exist_ok=True)

    # uncomment this to remove all old files first
    # for f in glob.glob('source/povcalnet/*.csv'):
    #     os.remove(f)
    if len(sys.argv) > 1:
        bracket = int(sys.argv[1])
        if bracket > 460:
            print("do not download bracket > 460")
            sys.exit(127)
        print(f'downloading {bracket}')
        process(bracket, resume=False)

    else:
        brackets = all_brackets[all_brackets < 2705]  # 2705 is in group 461
        run = partial(process, resume=False)
        with Pool(POOLSIZE) as p:
            r = range(len(brackets))
            p.map(run, r[::-1])
