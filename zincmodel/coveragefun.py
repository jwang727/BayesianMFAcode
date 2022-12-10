import numpy as np
import arviz as az

def calculatecoverage(samplemean, trace, availablechildstocksandflows, outputdatachildprocess):
    """
    function that calculates marginal posterior hdi and checks whether they contain the true value, for a single dataset
    """

    betastockssample = trace['posterior'].stocks
    betaflowsample = trace['posterior'].flows
    samplesoutput = np.hstack((betastockssample[0, :, :], betaflowsample[0, :, :]))
    samplemean = np.mean(samplesoutput, axis=0)

    ci_95 = az.hdi(trace, hdi_prob=0.95)

    ci_95_juststocks = az.hdi(trace['posterior'].stocks, hdi_prob=0.95)
    ci_95_justflows = az.hdi(trace['posterior'].flows, hdi_prob=0.95)

    ci_95_overall = np.vstack((ci_95_juststocks['stocks'], ci_95_justflows['flows']))
    truevalues = np.array(outputdatachildprocess['quantity'][availablechildstocksandflows])
    truevalues = np.reshape(truevalues, (truevalues.size, 1))
    ci_95_overall_with_true = np.hstack((ci_95_overall, truevalues))

    checkcoverage = (ci_95_overall_with_true[:, 0] <= ci_95_overall_with_true[:, 2]) & (
                ci_95_overall_with_true[:, 2] <= ci_95_overall_with_true[:, 1])
    checkcoverage = np.reshape(checkcoverage, (checkcoverage.size, 1))



    availableflownumbers = np.array(outputdatachildprocess['Flownumber'][availablechildstocksandflows])

    availableflownumbers = np.reshape(availableflownumbers, (availableflownumbers.size, 1))

    ci_95_marginal = np.hstack(
        (availableflownumbers, ci_95_overall_with_true, checkcoverage, np.reshape(samplemean, (samplemean.size, 1))))

    sampledistancestomean = [0] * samplesoutput.shape[0]
    for i in range(0, samplesoutput.shape[0]):
        sampledistancestomean[i] = np.linalg.norm(samplesoutput[i, :] - samplemean)

    cr_95_quantile = np.percentile(sampledistancestomean, 95)

    distancetotruth = np.linalg.norm(
        np.array(outputdatachildprocess['quantity'][availablechildstocksandflows]) - samplemean)

    return cr_95_quantile, distancetotruth, ci_95_marginal