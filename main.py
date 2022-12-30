##################################
# importing libraries
#################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, ttest_ind
from helpers.eda import *
from helpers.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 10000)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###############################
# Data Preparation and Analysis
###############################

A_ = pd.read_excel("ab_testing.xlsx", sheet_name="Control Group")
B_ = pd.read_excel("ab_testing.xlsx", sheet_name="Test Group")

check_df(A_)
check_df(B_)

check_outlier(A_, "Purchase")
check_outlier(B_, "Purchase")

""""
The success criterion for Bombomba.com is the Number of Purchases. 
For this reason, we will examine the Procurement variable. 
There are no missing values in either dataset. 
Also, there are no outliers in the Number of Purchases.
-----> Everything is going well...
"""

# Merge the two data sets
A_.columns = [i+"_A" for i in A_.columns]
B_.columns = [i+"_B" for i in B_.columns]
df = A_.merge(B_, left_index=True, right_index=True)
#df = pd.concat([A_, B_], axis=1)

#########################
# Implementing A/B Test
#########################

"""
1. Establish Hypotheses
2. Assumption Check
 - 1. Normality Assumption
 - 2. Variance Homogeneity (also do preprocessing, exploratory data analysis here if needed)
3. Implementation of the Hypothesis
 - 1. Independent two-sample t-test (parametric test) if assumptions are met
 - 2. Mannwhitneyu test if assumptions are not provided (non-parametric test)
4. Interpret results based on p-value
Note:
 - If normality is not provided, direct number 2. (do non-parametric test)
 - If normality is provided but variance homogeneity is not provided, an argument is entered for number 1.
 (Use parametric test, but "variance homogeneity is not provided" is entered as an argument.)
 - It can be useful to perform outlier analysis and correction before normality analysis.
 """

# 1. Establish Hypotheses

# H0: M1 = M2
# There is no statistical difference between the average purchase earned,
# by the maximum bidding strategy and the average purchase achieved by the average bidding strategy.

# H1: M1 != M2
# There is a statistical difference between the average purchase earned,
# by the maximum bidding strategy and the average purchases earned by the average bidding strategy.

# Did the difference between the possibilities of the two groups come about by chance?
df["Purchase_A"].mean()  # Mean of purchase of control group 550.894
df["Purchase_B"].mean()  # Mean of purchase of test group582.106


# 2. Assumption Check

# 2.1 Normality Assumption
# H0: There is no statistically significant difference between the std. normal distribution and the sample normal distribution
# H1: There is a statistically significant difference between the Std. normal distribution and the sample normal distribution.

# The test rejects the hypothesis of normality when the p-value is less than or equal to 0.05.
# We do not want to reject the null hypothesis in the tests that might be considered for assumptions.
# p-value < 0.05 (H0 rejected)
# p-value > 0.05 (H0 not rejected)

# Shapiro-Wilk Test
test_stat, pvalue = shapiro(df["Purchase_A"])
print("Test Statistic: %.4f, p-value: %.4f" % (test_stat, pvalue))
# p-value > 0.05 (H0 not rejected)

test_stat, pvalue = shapiro(df["Purchase_B"])
print("Test Statistic: %.4f, p-value: %.4f" % (test_stat, pvalue))
# p-value > 0.05 (H0 not rejected)

# 2.2 Variance Homogeneity Assumption
# H0: Variances are homogeneous
# H1: Variances are not homogeneous

# Levene Test
test_stat, pvalue = levene(df["Purchase_A"], df["Purchase_B"])
print("Test Statistic: %.4f, p-value: %.4f" % (test_stat, pvalue))
# p-value > 0.05 (H0 not rejected)


# 3. Implementation of the Hypothesis

# Since both assumptions are satisfied, Independent two-sample t-test (parametric test) is performed.
test_stat, pvalue = ttest_ind(df["Purchase_A"], df["Purchase_B"], equal_var=True)
print("Test Statistic: %.4f, p-value: %.4f" % (test_stat, pvalue))
# Test Statistic: -0.9416, p-value: 0.3493
# p-value > 0.05 (H0 not rejected)


# 4. Interpret results based on p-value

# p-value > 0.05 (H0 not rejected)
# There is no statistical difference between the average purchase earned,
# by the maximum bidding strategy and the average purchase achieved by the average bidding strategy.

# The difference in the purchase metric of the two groups occurred independently of the bidding strategy.
df["Purchase_A"].mean()  # Mean of purchase of control group 550.894
df["Purchase_B"].mean()  # Mean of purchase of test group582.106
