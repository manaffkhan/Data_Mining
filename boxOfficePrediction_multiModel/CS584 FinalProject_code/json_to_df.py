# a crude program to convert all the json with 10k movies each into one dataframe

import pandas as pd
import json

df = pd.DataFrame()
with open('0-10.json', 'r') as f:
    data = json.load(f)
    print(len(data))
    df = df.append(data)
    print(df.shape)
with open('10-20.json', 'r') as f:
    data = json.load(f)
    print(len(data))
    df = df.append(data)
    print(df.shape)
with open('20-30.json', 'r') as f:
    data = json.load(f)
    print(len(data))
    df = df.append(data)
    print(df.shape)
with open('30-40.json', 'r') as f:
    data = json.load(f)
    print(len(data))
    df = df.append(data)
    print(df.shape)
with open('40-50.json', 'r') as f:
    data = json.load(f)
    print(len(data))
    df = df.append(data)
    print(df.shape)
with open('50-60.json', 'r') as f:
    data = json.load(f)
    print(len(data))
    df = df.append(data)
    print(df.shape)
with open('60-70.json', 'r') as f:
    data = json.load(f)
    print(len(data))
    df = df.append(data)
    print(df.shape)
with open('70-80.json', 'r') as f:
    data = json.load(f)
    print(len(data))
    df = df.append(data)
    print(df.shape)
with open('80-90.json', 'r') as f:
    data = json.load(f)
    print(len(data))
    df = df.append(data)
    print(df.shape)

df.to_csv('final.csv')