import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub
from pygments.styles.dracula import orange

# Download latest version
# path = kagglehub.dataset_download("jsphyg/tipping")
#
# print("Path to dataset files:", path)

df = sns.load_dataset("tips")
print(df.head())
print()
print(df.isna().sum())
print()
print(df.describe())

fig, ax = plt.subplots(2,2,figsize=(12,10))
sns.histplot(df['total_bill'], ax=ax[0,0])
ax[0,0].set_title('Ratio of total bill')

sns.boxplot(data = df,x ='sex',y ='tip',hue='sex', ax=ax[0,1],legend=False,palette='Set2')
ax[0,1].set_title('Difference in gender')

sns.scatterplot(data = df ,x = 'total_bill',y = 'tip', hue = 'sex',ax=ax[1,0])
ax[1,0].set_title('The influence of gender and consumption on tipping')
ax[1,0].legend()

sns.boxplot(data = df,x ='smoker',y = 'tip',ax=ax[1,1],hue = 'smoker',legend=False,palette='Set2')
ax[1,1].set_title('The influence of smoker on tipping')
plt.suptitle('tips dataset analysis',fontsize=20,fontweight='bold')
plt.tight_layout()
plt.savefig('tips_dataset.png')
plt.show()