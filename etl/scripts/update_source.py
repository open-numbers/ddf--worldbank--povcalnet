import time
import pandas as pd
from ddf_utils.factory.common import download

# brackets data are downloaded from
# https://docs.google.com/spreadsheets/d/1QQqbMj6yYclwB3Q9xLYVNUFaJskYoZEQKpzYHvlnuas/edit#gid=0
brackets_csv = '../source/brackets.csv'
brackets_df = pd.read_csv(brackets_csv)
all_brackets = brackets_df['bracket_start'].to_list()
all_brackets.append(brackets_df['bracket_end'].iloc[-1])  # Append last value from bracket end

for i, bracket in enumerate(all_brackets):
    url = "http://iresearch.worldbank.org/PovcalNet/PovcalNetAPI.ashx?YearSelected=all&PovertyLine={}&Countries=all&display=C".format(bracket)
    file_csv = "../source/{:04d}.csv".format(i)
    download(url, file_csv, resume=False)
    time.sleep(2)
