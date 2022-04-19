#!/usr/bin/env python3

from csv import reader
import numpy as np

with open("complexity_ds_en.csv", "r") as f:
    data = list(reader(f.readlines()[1:]))
    data = [
        {
            "sent": line[1],
            "ratings": [int(x) for x in line[2:]]
        }
        for line in data
    ]
    data = [
        line for line in data
        if np.average(line["ratings"]) <= 3 or np.average(line["ratings"]) >= 4
    ]
    print(len(data))
    print(data[0])