# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
bank=pd.read_csv(path)
categorical_var= bank.select_dtypes(include = 'object')
print(categorical_var)
print('='*50)
numerical_var= bank.select_dtypes(include = 'number')
print(numerical_var)



# code ends here


# --------------
# code starts here
banks= bank.drop(['Loan_ID'],axis=1)
print(banks.isnull().sum())
print('='*50)
bank_mode= banks.mode()
banks= banks.fillna('bank_mode')
print(banks.isnull().sum())
#code ends here


# --------------
# Code starts here
avg_loan_amount= banks.pivot_table(index=['Gender','Married','Self_Employed'],values=['LoanAmount'],aggfunc='mean')
print(avg_loan_amount)



# code ends here



# --------------
# co=de starts here





loan_approved_se = banks[(banks['Self_Employed']== 'Yes') & (banks['Loan_Status']=='Y')]['Loan_Status'].value_counts().sum()
loan_approved_nse = banks[(banks['Self_Employed']== 'No') & (banks['Loan_Status']=='Y')]['Loan_Status'].value_counts().sum()
percentage_se = loan_approved_se/614*100
percentage_nse = loan_approved_nse/614*100
print(percentage_se)
print(percentage_nse)
# code ends here



# --------------
# code starts here

loan_term= banks['Loan_Amount_Term'].apply(lambda x: x/12)

big_loan_term= banks[loan_term >=25]['Loan_Amount_Term'].value_counts().sum()
print(loan_term)
print(big_loan_term)


# code ends here


# --------------
# code starts here
loan_groupby= banks.groupby('Loan_Status')
loan_groupby= loan_groupby['ApplicantIncome', 'Credit_History']
mean_values= loan_groupby.mean()
print(mean_values)



# code ends here


