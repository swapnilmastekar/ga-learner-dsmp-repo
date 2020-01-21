# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]

#Code starts here
data = pd.read_csv(path)
# sampling the dataframe
data_sample = data.sample(n=sample_size, random_state=0)
# finding the mean of the sample
sample_mean = data_sample.installment.mean()
print(sample_mean)
# finding the std of the sample
sample_std = data_sample.installment.std()
print(sample_std)
# finding the margin of error
margin_of_error = z_critical*(sample_std/math.sqrt(sample_size))
print(margin_of_error)
# finding the confidence interval
confidence_interval = (sample_mean - margin_of_error,sample_mean + margin_of_error)
print(confidence_interval)
# finding the true mean
true_mean = data.installment.mean()
print(true_mean)
# Code ends here





# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
fig, axes = plt.subplots(nrows = 3 , ncols = 1)
for i in range(len(sample_size)):
    m= []
    for j in range(1000):
        data_sample = data.sample(n=sample_size[i], random_state=0)
        m.append(data_sample.installment.mean())
    mean_series = pd.Series(m) 
    axes[i].hist(mean_series, bins =20,color = 'blue', edgecolor = 'black')
plt.show()


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
data['int.rate'] = data['int.rate'].str.rstrip('%')
data['int.rate'] = pd.to_numeric(data['int.rate'], errors='coerce')
data['int.rate'] = data['int.rate'].div(100)
z_statistic, p_value = ztest(x1=data[data['purpose']=='small_business']['int.rate'],value=data['int.rate'].mean(),alternative='larger')
print("Z-statistics = ",z_statistic)
print("p-value = ",p_value)

if p_value<0.05:
    inference = 'Reject'
else:
    inference = 'Accept'
    
print("We",inference, "the Null Hypothesis")


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
z_statistic, p_value = ztest(x1=data[data['paid.back.loan']=='No']['installment'],x2=data[data['paid.back.loan']=='Yes']['installment'])
print("Z-statistics = ",z_statistic)
print("p-value = ",p_value)

if p_value<0.05:
    inference = 'Reject'
else:
    inference = 'Accept'
    
print("We",inference, "the Null Hypothesis")



# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
# value counts of purpose when paid.back.loan in 'data' is Yes
yes = data[data['paid.back.loan']== 'Yes']['purpose'].value_counts()
# value counts of purpose when paid.back.loan in 'data' is No
no = data[data['paid.back.loan']== 'No']['purpose'].value_counts()
observed= pd.concat([yes.transpose(), no.transpose()],axis=1,keys= ['Yes','No'])
#applying chi-square test
chi2, p,dof,ex = chi2_contingency(observed)
print("chi2-statistics = ",chi2)
print("p-value = ",p)

if chi2 > critical_value:
    inference = 'Reject'
else:
    inference = 'Accept'
    
print("We",inference, "the Null Hypothesis")


