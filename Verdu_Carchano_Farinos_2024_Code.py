# ======================================================================================================================================================

# Code for the article Manuel Verdú, Óscar Carchano & José Emilio Farinós (2024) Arbitrage opportunities and event impacts on Spanish rights issues,
# Applied Economics Letters, 31:5, 401-421.

# ======================================================================================================================================================

# To execute this code, you must have the following files available in ‘https://doi.org/10.5281/zenodo.17897382’.
#   Verdu_Carchano_Farinos_2024_Data_Arbitrage_STES.csv
#   Verdu_Carchano_Farinos_2024_Data_LTES.csv

# Once the data is available in the same directory, you only need to execute the code to obtain the results shown in the article.

# The article can be found at: https://doi.org/10.1080/13504851.2022.2137293

# ======================================================================================================================================================

#################### LIBRARIES TO USE ####################

import os

import numpy as np                      # Allows to work with Series.
import pandas as pd						# Allows to organize the Data.
import statsmodels.formula.api as smf	# Allows to apply regression methods.

from scipy import stats                 # Allows to work with statistical tests and distributions

os.chdir('/Users/manuelverduhenares/Library/CloudStorage/Dropbox/1_Investigación/10_BBDD/2025_Verdu_Carchano_Farinós')

#################### START OF COMPLEMENTARY FUNCTIONS ####################

def calculate_stats(data, var_name, out_name, B = 0):
    """
    Calculate descriptive statistics and hypothesis tests
    """

    #Removing the atypical returns only for arbitrage strategy returns.

    if B == 0:
        data = data[data[out_name] == 0]
    
    values = data[var_name].dropna()
    n = len(values)

    values = values.reset_index(drop=True)    
    
    if n == 0:
        return {
            'N': 0,
            'Mean': np.nan,
            'Std': np.nan,
            'Median': np.nan,
            'T-test p-value': np.nan,
            'Wilcoxon p-value': np.nan,
            'Bootstrap CI Lower': np.nan,
            'Bootstrap CI Upper': np.nan
        }
    
    mean = values.mean()
    std = values.std()
    median = values.median()
    L = len(values)
    
    # T-test (testing if mean is different from 0)
    t_stat, t_pvalue = stats.ttest_1samp(values, 0)
    
    # Wilcoxon signed-rank test (testing if median is different from 0)
    try:
        wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(values)

    except:
        wilcoxon_pvalue = np.nan

    # Bootstrap test

    if B == 1:

        n_bootstrap = 999
        bootstrap_T = []
        np.random.seed(42)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(values, size=len(data), replace=True)
            bootstrap_mean = np.mean(bootstrap_sample)
            bootstrap_std = np.std(bootstrap_sample, ddof=1)
            bootstrap_t = bootstrap_mean / (bootstrap_std / np.sqrt(len(bootstrap_sample)))
            
            bootstrap_T.append(bootstrap_t)
        
        bootstrap_T = np.array(bootstrap_T)
        bootstrap_ci_lower = mean - np.percentile(bootstrap_T, 97.5) * (std / np.sqrt(len(values)))
        bootstrap_ci_upper = mean - np.percentile(bootstrap_T, 2.5) * (std / np.sqrt(len(values)))

    else:

        bootstrap_ci_lower = np.nan
        bootstrap_ci_upper = np.nan
    
    return {
        'N': n,
        'Mean': mean,
        'Std': std,
        'Median': median,
        'T-test p-value': t_pvalue,
        'Wilcoxon p-value': wilcoxon_pvalue,
        'Bootstrap CI Lower': bootstrap_ci_lower,
        'Bootstrap CI Upper': bootstrap_ci_upper
    }

#################### END OF COMPLEMENTARY FUNCTIONS ####################

#################### START OF THE CODE ####################

DAT = pd.read_csv('2025_Verdu_Carchano_Farinos_Data_Arbitrage_STES.csv', delimiter = ';') # Open the data file.
DAT = DAT.drop('#', axis = 1)

DAT2 = pd.read_csv('2025_Verdu_Carchano_Farinos_Data_LTES.csv', delimiter = ';') # Open the data file.
DAT2 = DAT2.set_index('Date')

##### RESULTS FROM THE ARBITRAGE STRATEGY #####

### Descriptive Statistics and Statistical Tests ###

## Loop for each of the results ##

for n in range(0, 12):

    if n == 0: # Average Net Return for Strategy I.

        series = 'RET1'
        outliers = 'OUT1'

    elif n < 11: # Net Return for the corresponding trading day.

        if n < 10:
            series = 'N0' + str(n)
        else:
            series = 'N' + str(n)

        outliers = 'OUN' + str(n)
 
    else: # Average Net Return for Strategy II.

        series = 'RET2'
        outliers = 'OUT2'

    print("=" * 80)
    print(f"ARBITRAGE RESULTS - DESCRIPTIVE STATISTICS AND HYPOTHESIS TESTS for {series}")
    print("=" * 80)
    
    ## Calculate statistics for different subsamples ##

    DATA_nO = DAT[DAT[outliers] == 0] #Removing the atypical returns.

    ## Total Sample ##

    stats_total = calculate_stats(DAT, series, outliers, 0)

    ## IBEX-35 firms ##

    data_ibe_high = DAT[DAT['IBEX'] == 1]
    stats_ibe_high = calculate_stats(data_ibe_high, series, outliers, 0)

    ## MC firms ##

    data_mc_high = DAT[DAT['IBEX'] == 0]
    stats_mc_high = calculate_stats(data_mc_high, series, outliers, 0)

    ## MAB firms ##

    data_mab_high = DAT[DAT['MAB'] == 1]
    stats_mab_high = calculate_stats(data_mab_high, series, outliers, 0)

    ## Dilutive Equity Offerings ##

    data_dil_high = DAT[DAT['DIL'] >= 0.5]
    stats_dil_high = calculate_stats(data_dil_high, series, outliers, 0)

    ## Non-Dilutive Equity Offerings ##

    data_dil_low = DAT[DAT['DIL'] < 0.5]
    stats_dil_low = calculate_stats(data_dil_low, series, outliers, 0)

    ## Monetary Equity Offerings ##

    data_din_1 = DAT[DAT['PRC'] != 1]
    stats_din_1 = calculate_stats(data_din_1, series, outliers, 0)

    ## Released Equity Offerings ##

    data_din_0 = DAT[DAT['PRC'] == 0]
    stats_din_0 = calculate_stats(data_din_0, series, outliers, 0)

    ## Insured Equity Offerings ##

    data_ins_1 = DAT[DAT['INS'] == 1]
    stats_ins_1 = calculate_stats(data_ins_1, series, outliers, 0)

    ## Non-Insured Equity Offerings ##

    data_ins_0 = DAT[DAT['INS'] == 0]
    stats_ins_0 = calculate_stats(data_ins_0, series, outliers, 0)

    print("\n" + "=" * 80)

    ## Summary table ##
    summary_stats = pd.DataFrame({
        'Total Sample': stats_total,
        'IBEX': stats_ibe_high,
        'MC': stats_mc_high,
        'MAB': stats_mab_high,
        'DIL >= 0.5': stats_dil_high,
        'DIL < 0.5': stats_dil_low,
        'MON = 1': stats_din_1,
        'REL = 0': stats_din_0,
        'INS = 0': stats_ins_1,
        'nINS = 0': stats_ins_0
    }).T

    print("\nSummary Table:")
    print(summary_stats)
    print("\n" + "=" * 80)
    
##### SHORT-TERM EVENT STUDY #####

## Loop for each of the events ##

EVE = ['ANN', 'MET', 'STR', 'END', 'RES', 'TRD']

for n in range(0, len(EVE)):

    series = EVE[n]

    print("=" * 80)
    print(f"SHORT-TERM EVENT STUDY - DESCRIPTIVE STATISTICS AND HYPOTHESIS TESTS for {series}")
    print("=" * 80)
    
    ## Calculate statistics for different subsamples ##

    DATA_nO = DAT[DAT[outliers] == 0] #Removing the atypical returns.

    ## Total Sample ##

    stats_total = calculate_stats(DAT, series, outliers, 1)

    ## IBEX-35 firms ##

    data_ibe_high = DAT[DAT['IBEX'] == 1]
    stats_ibe_high = calculate_stats(data_ibe_high, series, outliers, 1)

    ## MC firms ##

    data_mc_high = DAT[DAT['IBEX'] == 0]
    stats_mc_high = calculate_stats(data_mc_high, series, outliers, 1)

    ## MAB firms ##

    data_mab_high = DAT[DAT['MAB'] == 1]
    stats_mab_high = calculate_stats(data_mab_high, series, outliers, 1)

    ## Dilutive Equity Offerings ##

    data_dil_high = DAT[DAT['DIL'] >= 0.5]
    stats_dil_high = calculate_stats(data_dil_high, series, outliers, 1)

    ## Non-Dilutive Equity Offerings ##

    data_dil_low = DAT[DAT['DIL'] < 0.5]
    stats_dil_low = calculate_stats(data_dil_low, series, outliers, 1)

    ## Monetary Equity Offerings ##

    data_din_1 = DAT[DAT['PRC'] != 1]
    stats_din_1 = calculate_stats(data_din_1, series, outliers, 1)

    ## Released Equity Offerings ##

    data_din_0 = DAT[DAT['PRC'] == 0]
    stats_din_0 = calculate_stats(data_din_0, series, outliers, 1)

    ## Insured Equity Offerings ##

    data_ins_1 = DAT[DAT['INS'] == 1]
    stats_ins_1 = calculate_stats(data_ins_1, series, outliers, 1)

    ## Non-Insured Equity Offerings ##

    data_ins_0 = DAT[DAT['INS'] == 0]
    stats_ins_0 = calculate_stats(data_ins_0, series, outliers, 1)

    print("\n" + "=" * 80)

    ## Summary table ##
    summary_stats = pd.DataFrame({
        'Total Sample': stats_total,
        'IBEX': stats_ibe_high,
        'MC': stats_mc_high,
        'MAB': stats_mab_high,
        'DIL >= 0.5': stats_dil_high,
        'DIL < 0.5': stats_dil_low,
        'MON = 1': stats_din_1,
        'REL = 0': stats_din_0,
        'INS = 0': stats_ins_1,
        'nINS = 0': stats_ins_0
    }).T

    print("\nSummary Table:")
    print(summary_stats)
    print("\n" + "=" * 80)

##### LONG-TERM EVENT STUDY #####

portfolio = ['Total', 'IBEX', 'MC', 'MAB', 'DIL', 'nDIL', 'MON', 'REL', 'INS', 'nINS']

for n in range(0, len(portfolio)):

    Portfolio = portfolio[n]

    print("=" * 80)
    print(f"LONG-TERM EVENT STUDY - DESCRIPTIVE STATISTICS AND HYPOTHESIS TESTS for {Portfolio}")
    print("=" * 80)
    
    if Portfolio == 'MAB':
        Market = 'SmallCap'
    else:
        Market = 'IBEX35'

    model1 = Portfolio  + ' ~ ' + Market
    model2 = Portfolio  + ' ~ ' + Market + ' + SMB + HML'
    model3 = Portfolio  + ' ~ ' + Market + ' + SMB + HML + ILIQ'

    result1 = smf.ols(model1,DAT2).fit(cov_type='HC1')
    result2 = smf.ols(model2,DAT2).fit(cov_type='HC1')
    result3 = smf.ols(model3,DAT2).fit(cov_type='HC1')

    print(f"\n OLS Model results for: {model1}\n")
    print(result1.summary())
    print("\n" + "=" * 80)

    print(f"\n OLS Model results for: {model2}\n")
    print(result2.summary())
    print("\n" + "=" * 80)

    print(f"\n OLS Model results for: {model3}\n")
    print(result3.summary())
    print("\n" + "=" * 80)

#################### END OF THE CODE ####################