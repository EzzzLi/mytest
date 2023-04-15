import torch
import matplotlib.pyplot as plt
import numpy as np

label = torch.load('nosiy_labels_CIFAR100_swinv2_192_22k_250_160_SAM_e8.pth')
label = np.array(label)
label.sort()
print(label)

i = 0
while label[i] == -1:
    i = i + 1
nlabel = label[i:]
print(nlabel)

# table = [0 for i in range(100)]
# for i in label:
#     if i != -1:
#         table[i] = table[i] + 1

print(nlabel)
fig=plt.figure()
plt.hist(nlabel,bins=100)

plt.xlabel("Class")     # X轴标签
plt.ylabel("Number of Noisy Label")        # Y轴坐标标签
plt.title("Cifar100 epsilon=8 Noisy Label Distribution")      #  曲线图的标题
# 显示图形
plt.show()
fig.savefig('fig_swinv2_250_e8.png')