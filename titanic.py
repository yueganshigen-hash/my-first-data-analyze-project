import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/macbook/PycharmProjects/test/tested.csv', encoding='utf8')

print(df.head())
print()
print(df.describe())
print()
print(df.isna().sum())
print()
df['Age'] = df['Age'].fillna(df['Age'].mean())
print(df.head())
print()
print(df.isna().sum())
print()
# df = df.drop(columns='Cabin')
print(df.isna().sum())
df['Fare'] = df['Fare'].fillna(df['Fare'].interpolate())
print(df.isna().sum())
df.to_csv('/Users/macbook/PycharmProjects/test/tested.csv', index=False)

print(df['Pclass'].value_counts())
print()
print(df['Survived'].value_counts())
print()
print(df['Sex'].value_counts())
print()
print(df['Fare'].value_counts())
print()


fig, ax = plt.subplots(3, 2, figsize=(12, 8))
fig.suptitle('Titanic Survived')

df['Pclass'].value_counts().plot(kind='bar', ax=ax[0, 0], color='steelblue')
ax[0, 0].set_title('each class number')
ax[0, 0].set_xlabel('class')
ax[0, 0].set_ylabel('number')

df['Sex'].value_counts().plot(kind='pie', ax=ax[0, 1], color='orange', autopct='%1.1f%%')
ax[0, 1].set_title('Ratio of men and women')

df['Age'].value_counts().plot(kind='hist', ax=ax[1, 0], bins=20, color='green')
ax[1, 0].set_title('ratio of Age')
ax[1, 0].set_xlabel('Age')

df['Fare'].value_counts().plot(kind='hist', ax=ax[1, 1], bins = 20 , color='red')
ax[1, 1].set_title('ratio of Fare')
ax[1, 1].set_xlabel('Fare')

df.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=ax[2, 0], color=['yellow','green','red'])
ax[2, 0].set_title('Survived')
ax[2, 0].set_title('ratio of each class Survived')
ax[2, 0].set_xlabel('Survived')
ax[2, 0].set_xticks([0,1,2])
ax[2, 0].set_xticklabels(['first','second','third'],rotation=60)
s = df.groupby('Pclass')['Survived'].mean()
for i,v in enumerate(s):
    ax[2,0].text(i,v +0.01,f'{v:.1%}',ha='center')

df.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=ax[2,1], color=['pink','yellow'])
ax[2,1].set_title('ratio of each gender Survived')
ax[2,1].set_xlabel('Gender')
ax[2,1].set_xticks([0,1])
ax[2,1].set_xticklabels(['female','male'],rotation=60)
ax[2,1].set_ylabel('Survived')
s0 = df.groupby('Sex')['Survived'].mean()
for i,v in enumerate(s0):
    ax[2,1].text(i,v+0.01,f'{v:.1%}',ha='center')


plt.tight_layout()
plt.show()
