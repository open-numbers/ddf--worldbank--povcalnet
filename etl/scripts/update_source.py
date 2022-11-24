import os
import os.path as osp
import glob
import time
import numpy as np
from ddf_utils.factory.common import download
from multiprocessing import Pool
from functools import partial

# brackets data are downloaded from
# https://docs.google.com/spreadsheets/d/1QQqbMj6yYclwB3Q9xLYVNUFaJskYoZEQKpzYHvlnuas/edit#gid=0
# brackets_csv = '../source/fixtures/brackets.csv'
# brackets_df = pd.read_csv(brackets_csv)
# all_brackets = brackets_df['bracket_start'].to_list()
# all_brackets.append(
#     brackets_df['bracket_end'].iloc[-1])  # Append last value from bracket end

source_dir = '../source/income_mountain'
all_brackets = np.logspace(-7, 13, 201, endpoint=True, base=2)
# url_tmpl = "http://iresearch.worldbank.org/PovcalNet/PovcalNetAPI.ashx?YearSelected=all&PovertyLine={}&Countries=all&display=C&format=csv"
url_tmpl = "https://api.worldbank.org/pip/v1/pip?country=all&year=all&povline={}&fill_gaps=true&group_by=none&welfare_type=all&reporting_level=all&format=csv"
POOLSIZE = 5


def process(i, resume=True):
    bracket = all_brackets[i]
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
    # remove all old files first
    for f in glob.glob('../source/income_mountain/*.csv'):
        os.remove(f)
    run = partial(process, resume=False)
    with Pool(POOLSIZE) as p:
        r = range(len(all_brackets))
        p.map(run, r[::-1])
    print("done.")
