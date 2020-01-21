# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data = pd.read_csv(path)
data['Gender'].replace('-','Agender',inplace=True)
gender_count = data['Gender'].value_counts()
gender_count.plot(kind='bar')
plt.xlabel('Gender', fontsize=5)
plt.ylabel('No of count', fontsize=5)
plt.title('Gender Count')
plt.show()
#Code starts here 




# --------------
#Code starts here
alignment= data['Alignment'].value_counts()
plt.pie(alignment,autopct='%1.2f',startangle=90)
plt.show()


# --------------
#Code starts here
sc_df= data[['Strength','Combat']].copy()
sc_covariance= sc_df['Strength'].cov(sc_df['Combat'])
sc_strength= sc_df['Strength'].std()
sc_combat= sc_df['Combat'].std()
sc_pearson= sc_covariance/(sc_strength*sc_combat)
print('sc pearson = ',sc_pearson)
ic_df= data[['Intelligence','Combat']].copy()
ic_covariance= ic_df['Intelligence'].cov(ic_df['Combat'])
ic_intelligence= ic_df['Intelligence'].std()
ic_combat= ic_df['Combat'].std()
ic_pearson= ic_covariance/(ic_intelligence*ic_combat)
print('ic pearson = ',ic_pearson)


# --------------
#Code starts here
total_high = data['Total'].quantile(0.99)
super_best = data.loc[data['Total'] > total_high]
super_best_names = list(super_best['Name'])
print(super_best_names)


# --------------
#Code starts here
fig, (ax_1, ax_2,ax_3) = plt.subplots(1,3, figsize=(20,10))
ax_1.boxplot(super_best['Intelligence'])
ax_1.set_title('Intelligence')
ax_2.boxplot(super_best['Speed'])
ax_2.set_title('Speed')
ax_3.boxplot(super_best['Power'])
ax_3.set_title('Power')
plt.show()


