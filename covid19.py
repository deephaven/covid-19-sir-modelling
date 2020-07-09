import requests
from deephaven import *
from deephaven import TableTools
from deephaven import QueryScope
from deephaven import Plot
import theano.tensor as tt
import scipy.integrate as spi
import scipy as sc
import pymc3 as pm
import math as math
import numpy as np
import pandas as pd

##################################################################
######################## SIR SIMULATION ##########################
##################################################################

# defining a simulation based on SIR model
def SIR(init, time, beta, gamma):
    # beta - transmission rate [0,inf)
    # gamma - recovery probability per day [0,1]
    s = init[0] # number of susceptible
    i = init[1] # number of infected
    pop = s + i
    # computing derivatives
    dsdt = -(beta * i * s)/pop
    didt = (beta * i * s)/pop - gamma * i
    drdt = -(dsdt + didt)
    # return list of derivatives
    return [dsdt,didt,drdt]

def simulateSIR(init, params, days):
    # create time to simulate over
    time = np.linspace(1,days,days)
    # perform simulation with scipy
    sim = pd.DataFrame(spi.odeint(SIR, init, time, params))
    # adding column names
    sim.columns = ['S','I','R']
    return sim

##################################################################
###################### READING LIVE DATA #########################
##################################################################

# making dict of state populations
statePops = {'AK': 731545, 'AL': 4903185, 'AR': 3017804, 'AS': 49437, 'AZ': 7278717,
             'CA': 39512223, 'CO': 5758736, 'CT': 3565287, 'DC': 705749, 'DE': 973764,
             'FL': 21477737, 'GA': 10617423, 'GU': 168485, 'HI': 1415872, 'IA': 3155070,
             'ID': 1787065, 'IL': 12671821, 'IN': 6732219, 'KS': 2913314, 'KY': 4467673,
             'LA': 4648794, 'MA': 6892503, 'MD': 6045680, 'ME': 1344212, 'MI': 9986857,
             'MN': 5639632, 'MO': 6137428, 'MP': 51433, 'MS': 2976149, 'MT': 1068778,
             'NC': 10488084, 'ND': 762062, 'NE': 1934408, 'NH': 1359711, 'NJ': 8882190,
             'NM': 2096829, 'NV': 3080156, 'NY': 19453561, 'OH': 11689100, 'OK': 3956971,
             'OR': 4217737, 'PA': 12801989, 'PR': 3193694, 'RI': 1059361, 'SC': 5148714,
             'SD': 884659, 'TN': 6829174, 'TX': 28995881, 'UT': 3205958, 'VA': 8535519,
             'VI': 106235, 'VT': 623989, 'WA': 7614893, 'WI': 5822434, 'WV': 1792147,
             'WY': 578759}

# this function will unpack the contents of a URL
def saveUrl(url, outFile):
    r = requests.get(url, allow_redirects=True)
    if r.status_code != 200:
        raise Exception("Bad HTML request: status={} url={}".format(r.status_code, url))
    open(outFile, 'wb').write(r.content)

# this function will take in a csv and a name and return a dataframe with that name
def loadUSDf(file, valName):
    df = pd.read_csv(file)
    df = df.drop(columns=["pending", "hospitalizedCurrently", "hospitalizedCumulative", "inIcuCurrently", "inIcuCumulative",
        "onVentilatorCurrently", "onVentilatorCumulative", "dataQualityGrade", "lastUpdateEt", "dateModified", "checkTimeEt",
        "dateChecked", "totalTestsViral", "positiveTestsViral", "negativeTestsViral", "positiveCasesViral", "fips", "positiveIncrease",
        "negativeIncrease", "totalTestResults", "totalTestResultsIncrease", "posNeg", "deathIncrease", "hospitalizedIncrease",
        "hash", "commercialScore", "negativeRegularScore", "negativeScore", "positiveScore", "score", "grade"], axis=1)
    df = fillData(df)
    df['date'] = df['date'].apply(lambda d: str(d)[0:4] + '-' + str(d)[4:6] + '-' + str(d)[6:8])
    return df

# this function will take in a csv url and a name and return a Deephaven table from that csv 
def loadTable(url, valName, dfLoader):
    tmp_file = "/tmp/covid.csv"
    saveUrl(url, tmp_file)
    df = dfLoader(tmp_file, valName)
    return df#dataFrameToTable(df, convertUnknownToString=True).update("date = convertDateTime(date + `T12:00 NY`)")

# this function will attempt to fill in missing data from each state as accurately as possible
def fillData(df):
    df = df.sort_values(by=['date'])
    filledDf = []
    # loop through table for each state
    for state in np.unique(df.state):
        # subset by state
        thisDf = df.loc[df.state == state,]
        # fill NA forward with first good value
        thisDf = thisDf.fillna(method='ffill')
        thisDf.death = thisDf.death.fillna(0)
        # appending filled data to filledDf
        filledDf.append(thisDf)

    # concatenating filledDf into dataframe
    filledDf = pd.concat(filledDf)
    # adding recovered and death column to get removed
    filledDf['removed'] = filledDf.recovered.fillna(0) + filledDf.death.fillna(0)
    # return filled dataframe
    return filledDf

historicalStatesURL = 'https://covidtracking.com/api/v1/states/daily.csv'

stateData = loadTable(historicalStatesURL, 'stateData', loadUSDf)

stateDataTable = dataFrameToTable(stateData).update("date = convertDateTime(date + `T12:00 NY`)")

######################################################################
######################### MCMC FUNCTION ##############################
######################################################################

# this function will perform mcmc sampling on given data with specified initial conditions
def sampleMCMC(data, s0, i0):

    # splitting data into infections and time as numpy arrays
    dataRem = data['removed'].to_numpy()
    time = np.linspace(0,len(data)-1, len(data))

    # establishing model
    with pm.Model():
        # create beta, gamma priors
        beta = pm.Lognormal('beta', mu=0, sigma=.5)
        gamma = pm.Beta('gamma', alpha=2, beta=5)

        # observed data is modelled as a function of the above parameters
        sirRem = pm.Deterministic('sirRem',
            s0 + i0 - ((s0 + i0)**(beta/(beta - gamma)))*
            (s0 + i0*tt.exp(time*(beta - gamma)))**(-gamma/(beta - gamma)))

        obsRem = pm.TruncatedNormal('obsRem', mu=sirRem, sigma=sirRem/5000 + 1,
                                    lower=0, upper=s0+i0, observed=dataRem)

        # specifying model conditions
        step=pm.NUTS(target_accept=.9)
        start=pm.find_MAP()
        
        # execute sampling
        modelTrace = pm.sample(draws=5000, tune=5000, step=step, start=start, chains=8, cores=1)

    # return posterior samples and other information
    return modelTrace

###################################################################
######################### MCMC WITH DATA ##########################
###################################################################

# the below runs MCMC for every state, takes an eternity

# creating empty dictionary for mcmc samples
#mcmcSamples = {}

# looping through state data to compute MCMC for each state
#for state in np.unique(stateData.state):
#
#    try:
#        # subsetting data by state
#        thisState = stateData.loc[stateData.state == state,]
#        # creating initial conditions, assuming # infected starts at 10
#        s0 = statePops[state] - 10
#        i0 = 10
#        # executing mcmc sampling process for each state
#        thisMCMC = sampleMCMC(thisState, s0, i0)
#        # printing means
#        print(np.mean(thisMCMC['beta']))
#        print(np.mean(thisMCMC['gamma']))
#        # storing MCMC returns in dictionary
#        mcmcSamples.update({state: [thisMCMC['beta'], thisMCMC['gamma']]})
#    except Exception as e:
#        print(state + 'failed')
#        continue

#mcmcTable = createTableFromData(mcmcSamples)
#db.addTable('peters_namespace','firstSamples',mcmcTable)

# translates mcmc results into something I can plot with
firstSamples = db.t('peters_namespace', 'firstSamples')
firstSamplesDf = pd.DataFrame.transpose(tableToDataFrame(firstSamples))
firstSamplesDf.index.name = 'state'
firstSamplesDf.reset_index(inplace=True)
firstSamplesDf.columns = ['state', 'beta', 'gamma']
transformedSamples = firstSamplesDf.set_index(['state']).apply(pd.Series.explode).reset_index()
samplesTable = dataFrameToTable(transformedSamples)

#################################################################
############################ PLOTS ##############################
#################################################################

# data plot
stateSel = Plot.oneClick(stateDataTable, 'state')
dataRemPlot = (Plot.plot('state', stateSel, 'date','removed').
    chartTitle('Number of Removed over Time').
    xLabel('Time').
    yLabel('# of Removed').
    show())

# histograms of parameter distributions
stateSel2 = Plot.oneClick(samplesTable, 'state')
betaHist = (Plot.histPlot('state', stateSel2, 'beta', 12).
    chartTitle('MCMC Distribution of Beta').
    xLabel('Beta Value').
    yLabel('Frequency').
    show())

stateSel3 = Plot.oneClick(samplesTable, 'state')
gammaHist = (Plot.histPlot('state', stateSel3, 'gamma', 12).
    chartTitle('MCMC Distribution of Gamma').
    xLabel('Gamma Value').
    yLabel('Frequency').
    show())


# scatter plot of parameter density estimate for state
stateSel4 = Plot.oneClick(samplesTable, 'state')
densityPlot = (Plot.plot('density', stateSel4, 'beta', 'gamma').
    chartTitle('Parameter Density Estimate').
    plotStyle('scatter').
    xLabel('Beta Value').
    yLabel('Gamma Value').
    show())

# scatter plot of mean parameter estimates for all states
samplesMeans = samplesTable.avgBy('state')
meansPlot = (Plot.plot('means', samplesMeans, 'beta', 'gamma').
    chartTitle('Mean Parameter Estimates by State').
    plotStyle('scatter').
    xLabel('Beta Value').
    yLabel('Gamma Value').
    show())

# in order to use parameters to simulate SIR plot, since oneClick cannot be
# used to generate datasets but only to subset them, I must simulate SIR for every
# state that we have parameter values for and subset the resulting table with oneClick




# sir plot simulated with mcmc parameters
# setting parameters
initialSIR = [statePops['NH']-10,10,0]
params = (float(columnToNumpyArray(samplesMeans.where("state=`NH`").select("beta"), 'beta')),
    float(columnToNumpyArray(samplesMeans.where("state=`NH`").select("gamma"), 'gamma')))
days = 700
# executing simulation
sim = simulateSIR(initialSIR, params, days)
tableSim = dataFrameToTable(sim).update("Time = i")
#plotting results
simPlot = (Plot.plot("Susceptible", tableSim, 'Time', "S").
    plotStyle("Scatter").
    plot("Infected", tableSim, "Time", "I").
    plotStyle("Scatter").
    plot("Removed", tableSim, "Time", "R").
    plotStyle("Scatter").
    chartTitle("SIR Simulation").
    xLabel("Time").
    show())