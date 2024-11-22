import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
df = pd.read_excel('training_losses.xlsx')

# 打印数据检查
print("Data Head:")
print(df.head())

# 检查列名
print("\nColumns:")
print(df.columns)

# 检查数据类型
print("\nData Types:")
print(df.dtypes)

# 将 Epoch 和 Loss_Value 列转换为数值类型
df['Epoch'] = pd.to_numeric(df['Epoch'], errors='coerce')
df['Loss_Value'] = pd.to_numeric(df['Loss_Value'], errors='coerce')

# 检查转换后的数据类型
print("\nConverted Data Types:")
print(df.dtypes)

# 设置画布
plt.figure(figsize=(10, 6))

# 获取所有的Loss_Name
loss_names = df['Loss_Name'].unique()

# 遍历每个Loss_Name，绘制对应的Loss_Value
for loss_name in loss_names:
    subset = df[df['Loss_Name'] == loss_name]
    plt.plot(subset['Epoch'], subset['Loss_Value'], label=loss_name)

# 添加标题和标签
plt.title('Loss Values Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')

# 设置横坐标刻度以10个epoch为间隔
max_epoch = df['Epoch'].max()
plt.xticks(range(0, max_epoch + 1, 10))

# 添加图例
plt.legend()

# 保存图形
plt.savefig('loss_plot.png')

# 显示图形
plt.show()
