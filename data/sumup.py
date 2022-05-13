from importlib.metadata import distribution
import numpy as np
import pandas as pd

TIMES = 100

dict = {
    "Graph Type": [], 
    "Bid Type": [], 
    "Mechanism": [], 
    "Efficiency Ratio": [], 
    "Normalized Revenue": []
}
labels = list(filter(lambda x: len(x) > 0, (open('labels').read().split('\n'))))

data = np.zeros(len(labels) * 2)
for i in range(0, 10):
    filename = 'results' + str(i)
    datai = np.fromfile(filename, dtype=np.double)
    datai = datai.reshape(-1, TIMES).mean(axis = -1)
    data = data + datai
data = np.array(data / 10).reshape(-1, 2)

for i in range(0, len(labels)):
    names = list(filter(lambda x: len(x) > 0, labels[i].split(' ')))
    dict["Mechanism"].append(names[-1])
    dict["Bid Type"].append(names[-2])
    dict["Graph Type"].append(' '.join(names[:-2]))
    dict["Efficiency Ratio"].append(data[i][0])
    dict["Normalized Revenue"].append(data[i][1])

dataFrame = pd.DataFrame(dict)
dataFrame.to_csv('result.csv')