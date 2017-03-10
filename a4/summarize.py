"""
sumarize.py
"""
import json

def main():

    clusterfile = 'temp/clustersdata.txt'
    classifyfile = 'temp/classifysdata.txt'
    summaryfile = 'summary.txt'

    output = []

    with open(clusterfile, 'r') as f:
        for line in f:
            output.append(line)

    with open(classifyfile, 'r') as f:
        for line in f:
            output.append(line)

    with open(summaryfile, 'w') as f:
        for line in output:
            f.write(line)

if __name__ == '__main__':
    main()