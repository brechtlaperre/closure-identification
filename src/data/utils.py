import sys
from typing import NamedTuple
import pandas as pd

class DataSet(NamedTuple):
    x: pd.DataFrame
    y: pd.DataFrame
    bin_value: pd.DataFrame = None
    agyro: pd.DataFrame = None