import kagglehub
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt

# # Download latest version
# path = kagglehub.dataset_download("uciml/iris")
#
# print("Path to dataset files:", path)
# print(os.listdir(path))
print()
df = pd.read_csv(os.path.join('/Users/macbook/.cache/kagglehub/datasets/uciml/iris/versions/2', "Iris.csv"))
print(df.head())
print()
print(df.isna().sum())
print()

fig, ax = plt.subplots(2,2,figsize=(12,10))
sns.violinplot(data = df,x ='Species', y = 'PetalLengthCm',hue= 'Species',legend = False,palette = 'Set2',inner = 'quartitle',ax = ax[0,0])
ax[0,0].set_title('Petal length by Species')

corr = df.drop(columns='Species').corr()
sns.heatmap(corr,annot=True,fmt='.2f',
            cmap = 'coolwarm',ax = ax[0,1])
ax[0,1].set_xticklabels(labels=corr.columns,fontsize=8,rotation=45)
ax[0,1].set_title('Petal width by Species')

for sp in df['Species'].unique():
    sns.kdeplot(df[df['Species']==sp],
                ax = ax[1,0],
                x = 'PetalLengthCm',label = sp)
ax[1,0].set_title('Petal width by Species')
ax[1,0].legend()

sns.scatterplot(data = df,x='SepalLengthCm',y = 'SepalWidthCm',
                hue= 'Species',palette = 'Set2',ax = ax[1,1])
ax[1,1].set_title('Sepal width and Length by Species')

plt.tight_layout()
plt.savefig('SepalWidthLengthAndSepalLength.png')

plt.show()

