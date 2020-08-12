This program uses the most recent available data from [The Covid Tracking Project](https://covidtracking.com)
to estimate the transmission rate, removal rate, and mortality rate of COVID-19 in each state, using a model
called the SIR model.

This estimation is done via Markov chain Monte Carlo sampling through a Python package called PyMC3. We utilize
Deephaven through the Jupyter environment so that we can take advantage of PyMC3's useful plots, as well as
other Python plotting packages including Descartes.

For some of the states, the MCMC process is not successful, and which ones fail will change everytime your data
is updated. Some, such as Colorado, seem to fail consistently. There are a variety of reasons MCMC might fail,
including poor model fit or numerical instability.

The final plots demonstrate metrics such as overall COVID-19 risk by state, and provide the user the ability to
come to their own conclusions about the spread of COVID-19 in the US. Additionally, the probabilistic distributions
given by MCMC enable users to answer questions about the spread and mortality rates of COVID-19 in a probabilistic way,
which is often more useful than a point estimate.