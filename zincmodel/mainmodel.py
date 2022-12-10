import numpy as np
import pymc3 as pm
import arviz as az

from preprocessingfun import dataindices
from preprocessingfun import scale, sigmastd

from pymc3 import TruncatedNormal, Normal, MvNormal, HalfCauchy, InverseGamma, Model, glm, \
    plot_posterior_predictive_glm, sample, Lognormal


def truncatednormalmodel(priormean, covariancevec, designmatrixstockscompact, designmatrixflowscompact, \
                         datavector, dataflownumber, availablechildstocksandflows, m, findmap):

    """
    Function for main Bayesian model on zinc data
    """

    availablechildstocks = [i for i in availablechildstocksandflows if i < m]
    availablechildflows = [i for i in availablechildstocksandflows if i >= m]

    with Model() as model:

        betastocks = pm.Normal('stocks', mu=priormean[availablechildstocks],
                               sigma=np.sqrt(covariancevec[availablechildstocks]),
                               shape=priormean[availablechildstocks].shape[0])

        betaflows = pm.TruncatedNormal('flows', mu=priormean[availablechildflows],
                                       sigma=np.sqrt(covariancevec[availablechildflows]), lower=0.0,
                                       upper=2000.0 / scale,
                                       shape=priormean[availablechildflows].shape[0])

        stockindex, flowindex, CoMindex = dataindices(dataflownumber, m)


        likelihoodstocks = pm.Normal("stockdata", mu=pm.math.dot(designmatrixstockscompact[stockindex, :], betastocks)
                                                     + pm.math.dot(designmatrixflowscompact[stockindex, :], betaflows)
                                     , sigma=sigmastd, observed=datavector[stockindex])

        likelihoodflows = pm.TruncatedNormal("flowdata",
                                             mu=pm.math.dot(designmatrixstockscompact[flowindex, :], betastocks)
                                                + pm.math.dot(designmatrixflowscompact[flowindex, :], betaflows)
                                             , sigma=sigmastd, lower=0.0, observed=datavector[flowindex])

        likelihoodmassconserve = pm.Normal("CoM", mu=pm.math.dot(designmatrixstockscompact[CoMindex, :], betastocks)
                                                     + pm.math.dot(designmatrixflowscompact[CoMindex, :], betaflows)
                                           , sigma=sigmastd, observed=datavector[CoMindex])

    if findmap == 1:
        with model:
            modelmap = pm.find_MAP()
            mapoverall = np.hstack((modelmap['stocks'], modelmap['flows']))

        predictions = mapoverall

        trace = float("nan")

    else:
        with model:
            trace = sample(draws=10000, return_inferencedata=True, chains=1, init='jitter+adapt_diag', tune=2000,
                           target_accept=0.85, random_seed=123456)

        betastockssample = trace['posterior'].stocks
        betaflowsample = trace['posterior'].flows
        samplesoutput = np.hstack((betastockssample[0, :, :], betaflowsample[0, :, :]))
        samplemean = np.mean(samplesoutput, axis=0)

        predictions = samplemean

    return predictions, trace


def samplefromlikelihood(datamean, dataflownumber, numberofdraws, m):

    """
    Generate datasets from the likelihood
    """

    with Model() as model:

        stockindices, flowindices, CoMindices = dataindices(dataflownumber, m)


        likelihoodstocks = pm.Normal("stockdata", mu=datamean[stockindices]
                                     , sigma=sigmastd, shape=datamean[stockindices].shape[0])

        likelihoodflows = pm.TruncatedNormal("flowdata", mu=datamean[flowindices]
                                             , sigma=sigmastd, lower=0.0, shape=datamean[flowindices].shape[0])

        likelihoodmassconserve = pm.Normal("CoM", mu=datamean[CoMindices]
                                           , sigma=sigmastd, shape=datamean[CoMindices].shape[0])

        datatrace = sample(draws=numberofdraws, return_inferencedata=True, chains=1, init='adapt_full', tune=3000,
                           target_accept=0.8, random_seed=123456)

        betastockssample = datatrace['posterior'].stockdata
        betaflowsample = datatrace['posterior'].flowdata
        CoMsample = datatrace['posterior'].CoM
        samplesoutput = np.hstack((betastockssample[0, :, :], betaflowsample[0, :, :], 0 * CoMsample[0, :, :]))

    return samplesoutput