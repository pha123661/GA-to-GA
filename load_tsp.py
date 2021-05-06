import pickle
import os


def LoadTsp(path="./tsp_data"):
    '''
    Input: path to tsp datas
    Output: list of 2D numpy array
    '''
    ret = list()
    for filename in os.listdir(path):
        if filename.endswith(".pickle"):
            with open(os.path.join(path, filename), mode="rb") as file:
                ret.append(pickle.load(file))
    return ret


if __name__ == "__main__":
    pass
