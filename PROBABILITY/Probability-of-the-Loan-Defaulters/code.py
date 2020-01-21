# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
#Reading of the file
df = pd.read_csv(path)
#Calculate the probability p(A)for the event that fico credit score is greater than 700.
p_a = len(df[df['fico'] > 700])/len(df['fico'])
#Calculate the probabilityp(B) for the event that purpose == 'debt_consolation'
p_b = len(df[df['purpose'] == 'debt_consolidation'])/len(df['purpose'])
#Calculate the purpose == 'debt_consolidation'
df1 = df[df['purpose'] == 'debt_consolidation']
#Calculate the probablityp(B|A) for the event purpose == 'debt_consolidation' given 'fico' credit score is greater than 700 
p_a_b = (p_b * p_a)/p_a
print(p_a_b)
#check the independency
result = p_a_b == p_a
print(result)
# code enp_bds here


# --------------
# code starts here
#probability p(A) for the event that paid.back.loan == Yes
prob_lp = df[df['paid.back.loan'] == 'Yes'].shape[0]/df.shape[0]
print(prob_lp)
# probability p(B) for the event that credit.policy == Yes
prob_cs = df[df['credit.policy'] == 'Yes'].shape[0]/df.shape[0]
print(prob_cs)

new_df = df[df['paid.back.loan'] == 'Yes']
#probablityp(B|A) for the event paid.back.loan == 'Yes' given credit.policy == 'Yes' 
prob_pd_cs = new_df[new_df['credit.policy'] == 'Yes'].shape[0]/new_df.shape[0]
print(prob_pd_cs)
#conditional probability
bayes = (prob_pd_cs*prob_lp)/prob_cs
print(bayes)
# code ends here


# --------------
# code starts here

df1 = df[df['paid.back.loan'] == 'No']
df1['purpose'].value_counts().plot(kind='bar')
plt.show()
# code ends here


# --------------
# code starts here
#median for installment
inst_median = df['installment'].median()
# mean for installment
inst_mean = df['installment'].mean()
#plot the histogram for installment
df['installment'].plot(kind='hist')
plt.title("Probability Distribution of INSTALLMENT")
plt.axvline(inst_median, color = 'r', linestyle = 'dashed', linewidth = 2)
plt.axvline(inst_mean, color = 'g', linestyle = 'dashed', linewidth = 2)
plt.show()
#histogram for log anual income
mean = df['log.annual.inc'].mean()
median = df['log.annual.inc'].median()
df['log.annual.inc'].plot(kind='hist')
plt.title("Probability Distribution of log.annualincome")
plt.axvline(median, color = 'r', linestyle = 'dashed', linewidth = 2)
plt.axvline(mean, color = 'g', linestyle = 'dashed', linewidth = 2)
plt.show()
# code ends here


