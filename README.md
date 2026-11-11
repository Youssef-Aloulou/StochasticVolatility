# Simulated Likelihood Inference for Stochastic Volatility Models using Continuous Particle Filtering

> Implementation of the SV-GARCH particle filter and continuous-in-parameters likelihood estimation, with maximum likelihood on S&P 500 data.

Group project on stochastic volatility tracking through particle filtering.

This project is based on the research article:

> Pitt, M. K., Malik, S. and Doucet, A. (2014). Simulated likelihood inference for stochastic volatility models using continuous particle filtering. *Annals of the Institute of Statistical Mathematics*, 66(3), 527–552.
> DOI: https://doi.org/10.1007/s10463-014-0456-y
> Preprint (WRAP): http://wrap.warwick.ac.uk/63220/1/WRAP_Pitt_9971334-ec-080914-aism-d-13-00049final.pdf

We implement a particle filter for the SV-GARCH model, the continuous-in-parameters likelihood estimate described in the second part of the article, and likelihood maximization on real data (S&P 500).

### Data Used
**S&P 500** – daily index prices (`^GSPC`), downloaded with `yfinance` over the periods set in `src/sv_pf/config.py`.

---

## Environment and Dependencies

```bash
# Clone the repo, set up the environment, and install the local package
git clone git@github.com:Youssef-Aloulou/StochasticVolatility.git
cd StochasticVolatility
conda env create -f environment.yml

# Activate the environment
conda activate sv-pf

# Install the sv_pf package
pip install -e .

# Check that the main dependencies import correctly
python -c "import sv_pf, particles; print('OK')"
```

## Data
The data are daily closing prices of the S&P 500 index (`^GSPC`), downloaded from Yahoo Finance with `yfinance` (adjusted for dividends and splits).

Five periods are used. Four come from the empirical section of Pitt, Malik and Doucet (2014), so our estimates can be compared with the article's tables. The last one is our own choice.

| Name | Period | Article reference | Why this period |
|---|---|---|---|
| `1982-1989` | 02/02/1982 – 29/12/1989 | Table 6, Figure 2 | Main period of the article; contains a large shock around the middle of the sample |
| `1995-2003` | 16/05/1995 – 24/04/2003 | Table 7, Figure 3 | Covers the late 1990s and the dot-com period |
| `2000-2008` | 19/12/2000 – 12/12/2008 | Table 8, Figure 4 | Covers the dot-com aftermath and the 2008 financial crisis |
| `1987-2011` | 31/03/1987 – 13/01/2011 | Table 9, Figure 5 | Longest series in the article, which also includes the 1987 crash |
| `2000-2020` | 01/01/2000 – 31/12/2020 | Not in the article | Our own test period: longer sample to check the stability of the estimator. Not directly comparable with the article's tables |

The periods are defined in `src/sv_pf/config.py` (`PERIODS`). To download them all:

```bash
# Download the S&P 500 daily prices for every period in PERIODS (one CSV per period in data/raw/)
python -m sv_pf.data

# Compute daily log-returns y_t = 100 * log(C_t / C_{t-1}) for every period in PERIODS (one CSV per period in data/processed/)
python -m sv_pf.preprocess

```

Files are saved in `data/raw/` as `sp500_<name>.csv` (for example `data/raw/sp500_1982-1989.csv`). They are not versioned.


## TODO

| # | Task | Where | Owner |
|---|---|---|---|
| 1 | SV-GARCH particle filter **Appendix B : Particle filter estimation of SV-GARCH model** (from scratch, no `particles`): estimate $\ell(\theta)$ at fixed $\theta$ with a standard bootstrap filter | `notebooks/04_pf_svgarch.ipynb` | |
| 2 | Smooth likelihood: same filter with continuous resampling, so $\theta \mapsto \ell(\theta)$ is smooth | `notebooks/05_continuous_pf.ipynb` | |
| 3 | MLE on the S&P 500 (BFGS on the continuous filter), comparison with the article | `notebooks/06_mle_sp500.ipynb` | |
| 4 | Report and results (AIC/BIC vs. the article) | `README.md` / report | |

**Already done:** SV / SVL / SVLJ models (with `particles`) and simulations (`notebooks/02_simulation_sv.ipynb`), SV-GARCH simulation (`notebooks/03_sv-garch.ipynb`).

## Authors

- Youssef Aloulou
- Younes Oudina
- Rou SENGHAK
- Marius Calaque