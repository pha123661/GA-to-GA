import pickle
import os
import numpy as np


def LoadTsp(path="./"):
    '''
    Input: path to tsp datas
    Output: list of 2D numpy array
    '''
    ret = list()
    for filename in os.listdir(path):
        if filename.endswith(".pickle"):
            with open(os.path.join(path, filename), mode="rb") as file:
                ret.append(np.array(pickle.load(file)))
    return ret


if __name__ == "__main__":
    print(LoadTsp(path="./tsp_data"))
