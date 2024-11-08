import re
import matplotlib.pyplot as plt

# 初始化列表来存储数据
epochs = []
learning_rates = []
D_A_values = []
G_A_values = []
cycle_A_values = []
idt_A_values = []
D_B_values = []
G_B_values = []
cycle_B_values = []
idt_B_values = []

# 读取文件并解析数据
with open('train.log', 'r') as file:
    for line in file:
        # 匹配 epoch 信息
        epoch_match = re.match(r'End of epoch (\d+) / 200', line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            epochs.append(epoch)
            continue
        
        # 匹配 learning rate 信息
        lr_match = re.match(r'learning rate = (\d+\.\d+)', line)
        if lr_match:
            learning_rate = float(lr_match.group(1))
            learning_rates.append(learning_rate)
            continue
        
        # 匹配 D_A, G_A, cycle_A, idt_A, D_B, G_B, cycle_B, idt_B 信息
        value_match = re.match(r'(D_A|G_A|cycle_A|idt_A|D_B|G_B|cycle_B|idt_B): (\d+\.\d+)', line)
        if value_match:
            key = value_match.group(1)
            value = float(value_match.group(2))
            if key == 'D_A':
                D_A_values.append(value)
            elif key == 'G_A':
                G_A_values.append(value)
            elif key == 'cycle_A':
                cycle_A_values.append(value)
            elif key == 'idt_A':
                idt_A_values.append(value)
            elif key == 'D_B':
                D_B_values.append(value)
            elif key == 'G_B':
                G_B_values.append(value)
            elif key == 'cycle_B':
                cycle_B_values.append(value)
            elif key == 'idt_B':
                idt_B_values.append(value)

# 确保所有列表的长度一致
max_length = len(epochs)
for i in range(max_length):
    if i >= len(learning_rates):
        learning_rates.append(None)
    if i % 10 != 0:
        D_A_values.insert(i, None)
        G_A_values.insert(i, None)
        cycle_A_values.insert(i, None)
        idt_A_values.insert(i, None)
        D_B_values.insert(i, None)
        G_B_values.insert(i, None)
        cycle_B_values.insert(i, None)
        idt_B_values.insert(i, None)

# 创建一个包含多个子图的画布
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Metrics vs Epoch', fontsize=16)

# 绘制 learning rate 图
axes[0, 0].plot(epochs, learning_rates, marker='o')
axes[0, 0].set_title('Learning Rate vs Epoch')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Learning Rate')
axes[0, 0].set_xticks(range(0, max(epochs) + 1, 10))
axes[0, 0].grid(True)

# 绘制其他图
metrics = {
    'D_A': D_A_values,
    'G_A': G_A_values,
    'cycle_A': cycle_A_values,
    'idt_A': idt_A_values,
    'D_B': D_B_values,
    'G_B': G_B_values,
    'cycle_B': cycle_B_values,
    'idt_B': idt_B_values
}

# 将每个指标的图放在不同的子图中
for i, (metric, values) in enumerate(metrics.items()):
    row = i // 3
    col = i % 3
    axes[row, col].plot(epochs, values, marker='o')
    axes[row, col].set_title(f'{metric} vs Epoch')
    axes[row, col].set_xlabel('Epoch')
    axes[row, col].set_ylabel(metric)
    axes[row, col].set_xticks(range(0, max(epochs) + 1, 10))
    axes[row, col].grid(True)

# 调整子图之间的间距
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存整个画布
plt.savefig('combined_metrics.png')
plt.show()