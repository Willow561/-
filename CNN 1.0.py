import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

def evaluate_model(model, X, y, target_names, n_splits=5, n_repeats=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []

    for repeat in range(n_repeats):
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            y_train_pred_classes = np.argmax(y_train_pred, axis=1)
            y_test_pred_classes = np.argmax(y_test_pred, axis=1)
            y_train_true_classes = np.argmax(y_train, axis=1)
            y_test_true_classes = np.argmax(y_test, axis=1)

            train_acc = accuracy_score(y_train_true_classes, y_train_pred_classes)
            test_acc = accuracy_score(y_test_true_classes, y_test_pred_classes)
            train_f1 = f1_score(y_train_true_classes, y_train_pred_classes, average='weighted')
            test_f1 = f1_score(y_test_true_classes, y_test_pred_classes, average='weighted')

            accuracies.append(test_acc)
            f1_scores.append(test_f1)

            print(f'{target_names} - Repeat {repeat+1}, Fold {len(accuracies)} - Test Accuracy: {test_acc:.4f}, Test F-measure: {test_f1:.4f}')

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    print(f'{target_names} - Mean Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}')
    print(f'{target_names} - Mean Test F-measure: {mean_f1:.4f} ± {std_f1:.4f}')

    return accuracies, f1_scores

def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 构建模型架构图
input_shape = (28, 28, 1)  # 示例输入形状
num_classes = 10  # 示例类别数
model = create_cnn_model(input_shape, num_classes)
plot_model(model, to_file='model_CNN.png', show_shapes=True, show_layer_names=True)

# 示例数据
X = np.random.rand(1000, 28, 28, 1)
y = np.random.randint(0, 10, size=(1000, 10))

# 评估模型
evaluate_model(model, X, y, 'Example')

# 定义不同的模型
model_ethnicity = create_cnn_model(input_shape, num_classes)
model_gender = create_cnn_model(input_shape, num_classes)
model_age = create_cnn_model(input_shape, num_classes)

# 示例训练和测试数据
X_train_ethnicity = np.random.rand(800, 28, 28, 1)
y_train_ethnicity = np.random.randint(0, 10, size=(800, 10))
X_test_ethnicity = np.random.rand(200, 28, 28, 1)
y_test_ethnicity = np.random.randint(0, 10, size=(200, 10))

X_train_gender = np.random.rand(800, 28, 28, 1)
y_train_gender = np.random.randint(0, 10, size=(800, 10))
X_test_gender = np.random.rand(200, 28, 28, 1)
y_test_gender = np.random.randint(0, 10, size=(200, 10))

X_train_age = np.random.rand(800, 28, 28, 1)
y_train_age = np.random.randint(0, 10, size=(800, 10))
X_test_age = np.random.rand(200, 28, 28, 1)
y_test_age = np.random.randint(0, 10, size=(200, 10))

# 评估不同模型
acc_ethnicity, f1_ethnicity = evaluate_model(model_ethnicity, X_train_ethnicity, y_train_ethnicity, 'Ethnicity')
acc_gender, f1_gender = evaluate_model(model_gender, X_train_gender, y_train_gender, 'Gender')
acc_age, f1_age = evaluate_model(model_age, X_train_age, y_train_age, 'Age')

# 可视化评估结果
labels = ['Ethnicity', 'Gender', 'Age']
acc_means = [np.mean(acc_ethnicity), np.mean(acc_gender), np.mean(acc_age)]
acc_stds = [np.std(acc_ethnicity), np.std(acc_gender), np.std(acc_age)]
f1_means = [np.mean(f1_ethnicity), np.mean(f1_gender), np.mean(f1_age)]
f1_stds = [np.std(f1_ethnicity), np.std(f1_gender), np.std(f1_age)]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, acc_means, width, label='Accuracy', yerr=acc_stds, color='skyblue')
rects2 = ax.bar(x + width/2, f1_means, width, label='F1 Score', yerr=f1_stds, color='lightgreen')

ax.set_ylabel('Scores')
ax.set_title('Model Evaluation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()