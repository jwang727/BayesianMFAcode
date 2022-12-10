import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

scale=100.0
sigmastd=1.0

def preprocessingzinc(datastocks, dataflows):
    """
    Function for preprocessing zinc data
    """

    datastocks['quantity'] = datastocks['quantity']/scale
    dataflows['quantity'] = dataflows['quantity']/scale

    datachildstocks = datastocks.loc[datastocks['ParentProcess'] == 0]

    childstocks = datachildstocks.Process.unique()

    m = len(childstocks)

    N = m + m * m

    Flownumberfromvector = list(range(0, m))

    for i in range(0, m):
        Flownumberfromvector = Flownumberfromvector + [i] * m

    Flownumbertovector = ['nan'] * m
    for i in range(0, m):
        Flownumbertovector = Flownumbertovector + list(range(0, m))

    Flownumberfromvector = np.reshape(Flownumberfromvector, (len(Flownumberfromvector), 1))

    Flownumbertovector = np.reshape(Flownumbertovector, (len(Flownumbertovector), 1))



    allflownumbersmatrix = np.hstack(
        ((np.reshape(list(range(0, N)), (N, 1)), Flownumberfromvector, Flownumbertovector)))


    processnamesdict = pd.Series(datachildstocks.Process.values,
                                 index=datachildstocks.Processnumber.astype(str)).to_dict()

    processnamesdict['nan'] = 'Stock'



    dataflows["Flownumber"] = m + (dataflows["Flownumberfrom"]) * m + dataflows["Flownumberto"]

    datastocks.sort_values('Processnumber', inplace=True)
    dataflows.sort_values('Flownumber', inplace=True)

    availablestocksfull = np.copy(datastocks.Processnumber.unique())
    availableflowsfull = np.copy(dataflows.Flownumber.unique())
    availablestockdatafull = datastocks.quantity.to_numpy()
    availableflowdatafull = dataflows.quantity.to_numpy()
    availablestocksandflowsfull = np.concatenate((availablestocksfull, availableflowsfull))
    availablestockandflowdatafull = np.concatenate((availablestockdatafull, availableflowdatafull))


    availabledatafull = np.column_stack((availablestocksandflowsfull, availablestockandflowdatafull))

    availabledatafulldataframe = pd.DataFrame(
        {'Flownumber': availablestocksandflowsfull.astype(int), 'quantity': availablestockandflowdatafull})


    availabledatafulldict = pd.Series(availabledatafulldataframe.quantity.values,
                                      index=availabledatafulldataframe.Flownumber.astype(str)).to_dict()

    availabledatafulldataframenozerostocks = availabledatafulldataframe[availabledatafulldataframe['quantity'] != 0]
    availabledatafulldataframenozerostocksnona = availabledatafulldataframenozerostocks.dropna(subset=['quantity'])

    outputdatachildprocess = pd.DataFrame(
        {'Flownumber': allflownumbersmatrix[:, 0], 'Flownumberfrom': allflownumbersmatrix[:, 1],
         'Flownumberto': allflownumbersmatrix[:, 2]})
    outputdatachildprocess['From'] = outputdatachildprocess['Flownumberfrom'].astype(str).map(processnamesdict)
    outputdatachildprocess['To'] = outputdatachildprocess['Flownumberto'].astype(str).map(processnamesdict,
                                                                                          na_action='ignore')
    outputdatachildprocess['quantity'] = outputdatachildprocess['Flownumber'].map(availabledatafulldict)

    availabledatafullnozerostocksnona = availabledatafulldataframenozerostocksnona.to_numpy()
    availabledatafullnozerostocks = availabledatafulldataframenozerostocks.to_numpy()

    datamat = availabledatafullnozerostocksnona


    availablechildstocksandflows = availabledatafullnozerostocks[:, 0]

    availablechildstocksandflows = [int(x) for x in availablechildstocksandflows]
    availablechildstocksandflows = sorted(list(set(availablechildstocksandflows)))


    return datamat, outputdatachildprocess, availablechildstocksandflows, m, N


def createincompletedata(points, datamat, availablechildstocksandflows, m, N):
    """
    Function for creating a design matrix of subset of the data
    """

    designmatrix = np.empty((1, N))
    datavector = []

    dataflownumber = []

    for row in datamat:
        if np.isnan(row[1]) == False and row[1] != 0:
            newrow = np.zeros((1, N))
            if row[0] in points:
                newrow[:, int(row[0])] = 1
                datavector.append(row[1])
                designmatrix = np.vstack([designmatrix, newrow])

                dataflownumber.append(row[0])

    for i in range(0, m):
        newrow = np.zeros((1, N))
        newrow[:, int(i)] = 1

        for j in range(0, m):
            newrow[:, int(i + (j + 1) * m)] = -1
            newrow[:, int((i + 1) * m + j)] = 1

        datavector.append(0)
        designmatrix = np.vstack([designmatrix, newrow])

        dataflownumber.append(-1)

    designmatrix = np.delete(designmatrix, (0), axis=0)
    datavector = np.array(datavector)

    availablechildstocks = [i for i in availablechildstocksandflows if i < m]
    availablechildflows = [i for i in availablechildstocksandflows if i >= m]

    designmatrixcompact = designmatrix[:, availablechildstocksandflows]
    designmatrixstockscompact = designmatrix[:, availablechildstocks]
    designmatrixflowscompact = designmatrix[:, availablechildflows]

    return designmatrixcompact, designmatrixstockscompact, designmatrixflowscompact, datavector, dataflownumber


def dataindices(dataflownumber, m):
    """
    Function for retrieving indices of the data that belongs to stock or flow data, or conservation of mass
    """

    stockindex = [i for i in range(len(dataflownumber)) if 0 <= dataflownumber[i] < m]
    flowindex = [i for i in range(len(dataflownumber)) if dataflownumber[i] >= m]
    CoMindex = [i for i in range(len(dataflownumber)) if dataflownumber[i] == -1]

    return stockindex, flowindex, CoMindex