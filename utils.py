import csv
import numpy as np

def find_acc():
    fp = open("./accuracy_2.txt", "r")
    lines = csv.reader(fp)
    acc = []
    for i in lines:
        acc.append(float(i[0]))

    final_acc = []
    for i in range(15):
        final_acc.append(np.mean([acc[i], acc[i + 15]]))
    print(final_acc)

find_acc()
