import time
import numpy as np
from ddf_utils.factory.common import download
from multiprocessing import Pool

# brackets data are downloaded from
# https://docs.google.com/spreadsheets/d/1QQqbMj6yYclwB3Q9xLYVNUFaJskYoZEQKpzYHvlnuas/edit#gid=0
# brackets_csv = '../source/fixtures/brackets.csv'
# brackets_df = pd.read_csv(brackets_csv)
# all_brackets = brackets_df['bracket_start'].to_list()
# all_brackets.append(
#     brackets_df['bracket_end'].iloc[-1])  # Append last value from bracket end

all_brackets = np.logspace(-7, 13, 201, endpoint=True, base=2)
url_tmpl = "http://iresearch.worldbank.org/PovcalNet/PovcalNetAPI.ashx?YearSelected=all&PovertyLine={}&Countries=all&display=C&format=csv"
POOLSIZE = 5


def process(i):
    bracket = all_brackets[i]
    url = url_tmpl.format(bracket)
    file_csv = "../source/{:04d}.csv".format(i)
    print(file_csv)
    download(url, file_csv, resume=True, progress_bar=False, backoff=5, timeout=60)
    time.sleep(5)


if __name__ == "__main__":
    with Pool(POOLSIZE) as p:
        r = range(len(all_brackets))
        p.map(process, r[::-1])
    print("done.")
