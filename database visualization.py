import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 示例数据加载
df = pd.read_csv('D:/age_gender.csv')

# 基本数据探索
def basic_eda(df):
    print("\n Number of null values: ")
    print(df.isnull().sum())
    print("\n\n --------- ")
    print("\n Value count of age: ")
    print(df['age'].value_counts())
    print("\n\n --------- ")
    print("\n Value count of ethnicity: ")
    print(df['ethnicity'].value_counts())
    print("\n\n --------- ")
    print("\n Value count of gender: ")
    print(df['gender'].value_counts())

basic_eda(df)

# 显示特征和目标变量的前几行数据
X = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
y = df[['age', 'ethnicity', 'gender']]
print(X.head())
print(y.head())

# 显示目标变量的唯一值数量
print(y.nunique())

# 计算图像像素数、图像高度和宽度
num_pixels = len(X.iloc[0])
img_height = int(np.sqrt(num_pixels))
img_width = int(np.sqrt(num_pixels))
print(num_pixels, img_height, img_width)

# 将像素数据转换为图像格式
X = np.array(X.tolist()).reshape(-1, img_height, img_width, 1)

# reshape data
X = X.reshape(-1, 48, 48, 1)
print("X shape: ", X.shape)

# 数据可视化
plt.figure(figsize=(16, 16))
for i, a in zip(np.random.randint(0, len(X), 25), range(1, 26)):
    plt.subplot(5, 5, a)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i].reshape(48, 48), cmap='gray')
    plt.xlabel(
        "Age: " + str(y['age'].iloc[i]) +
        " Ethnicity: " + str(y['ethnicity'].iloc[i]) +
        " Gender: " + str(y['gender'].iloc[i])
    )
plt.show()

# 进一步数据可视化
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='gender', data=df)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='ethnicity', data=df)
plt.title('Ethnicity Distribution')
plt.xlabel('Ethnicity')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='gender', y='age', data=df)
plt.title('Age Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()

# 生成相关性矩阵图
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()