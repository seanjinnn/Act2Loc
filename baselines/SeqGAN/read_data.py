# czc
import pandas as pd
import numpy as np

path = "real.data"
data = np.loadtxt(path, delimiter=",")

def initData():
    path = "real.data"
    df = pd.read_csv(path, header=None, sep='\s+')
    print(len(df))

    np_tests = np.array(df)

    for i, test in enumerate(np_tests):
        print("第%d行的数据是%s:" % (i, test))