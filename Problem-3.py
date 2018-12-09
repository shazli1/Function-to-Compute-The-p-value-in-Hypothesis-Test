import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
from scipy.stats import norm

# Read data files
data_3_1 = np.loadtxt(r'C:\Users\chtv2985\Desktop\Assig-2\Data\Data3-1.txt', delimiter=',')
data_3_2 = np.loadtxt(r'C:\Users\chtv2985\Desktop\Assig-2\Data\Data3-2.txt', delimiter=',')
data_3_3 = np.loadtxt(r'C:\Users\chtv2985\Desktop\Assig-2\Data\Data3-3.txt', delimiter=',')

sig_level = 0.05

######################################################################################
# Function to calculate pvalue & boundaries of acceptance region of two input datasets
######################################################################################
def calc_pvalue(dataset1, dataset2, alpha):
    # Calculate Mean, Variance, and Standard Deviation
    mean1 = np.mean(dataset1)
    mean2 = np.mean(dataset2)

    var1 = np.var(dataset1)
    var2 = np.var(dataset2)

    stand_dev1 = math.sqrt(var1)
    stand_dev2 = math.sqrt(var2)

    # Normalizing the distribution of Sample Dataset "i.e: dataset2 in this function"
    zscore = (mean2 - mean1) / (stand_dev1 / math.sqrt(len(dataset2)))

    pvalue = 2 * (1 - norm.cdf(zscore))    # Calculate pvalue bases on Z

    # To get the boundaries of the acceptance region
    # Get Z based on alpha = 0.05
    # But as we are using standard normal distribution, we will use alpha = 0.025 as it is symmetric around 0
    z_based_on_alpha = norm.ppf(1 - (alpha/2))
    # Calculate (x_bar - mean) = Z * (std_dev/sqrt(n))
    mean_diff = z_based_on_alpha * (stand_dev1 / math.sqrt(len(dataset2)))
    accept_boundary_1 = mean1 - mean_diff
    accept_boundary_2 = mean1 + mean_diff

    return zscore, pvalue, accept_boundary_1, accept_boundary_2
    #print("=======================================")


# Calculate Zscore & Pvalue of Data3-1 & Data3-2
################################################
z_1_2, pvalue_1_2, boundary_a_1_2, boundary_b_1_2 = calc_pvalue(data_3_1, data_3_2, sig_level)
print("Zscore of Data3-1 & Data3-2: %s" % (z_1_2))
print("Pvalue of Data3-1 & Data3-2: %s" % (pvalue_1_2))
print("Boundaries of The Acceptance Region: %s" % boundary_a_1_2, boundary_b_1_2)

# Cross-checking the function performance with python function
print("Pvalue Obtained From Python Function:")
print(str(scipy.stats.norm.sf(abs(z_1_2))*2))
print("================================================")


# Calculate Zscore & Pvalue of Data3-1 & Data3-3
#################################################
z_1_3, pvalue_1_3, boundary_a_1_3, boundary_b_1_3 = calc_pvalue(data_3_1, data_3_3, sig_level)
print("Zscore of Data3-1 & Data3-3: %s" % (z_1_3))
print("Pvalue of Data3-1 & Data3-3: %s" % (pvalue_1_3))
print("Boundaries of The Acceptance Region: %s" % boundary_a_1_3, boundary_b_1_3)

# Cross-checking the function performance with python function
print("Pvalue Obtained From Python Function:")
print(str(scipy.stats.norm.sf(abs(z_1_3))*2))
print("================================================")

#print(np.mean(data_3_1), np.mean(data_3_2), np.mean(data_3_3))
#print(len(data_3_1), len(data_3_2), len(data_3_3))


# Plot the Histogram of Data3-1 & Data3-2
bins = 50
plt.hist(data_3_1, bins, label='Data3-1')
plt.hist(data_3_2, bins, label='Data3-2')
plt.legend(loc='upper right')
plt.show()

# Plot the Histogram of Data3-1 & Data3-3
plt.hist(data_3_1, bins, label='Data3-1')
plt.hist(data_3_3, bins, label='Data3-3')
plt.legend(loc='upper right')
plt.show()