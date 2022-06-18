import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

train_losses = []
test_losses = []

path = 'loss_plot 2.txt'
i = 0
with open(path, 'r', encoding="utf-8") as f:
    datafile = f.readlines()
    datafile = [x[x.index('loss = '):] if 'loss' in x else x for x in datafile]
    for line in datafile:
        line = line.split('loss = ')[-1]
        if i % 2 == 0:
            train_losses.append(float(line))
        else:
            test_losses.append(float(line))
        i += 1

# for i in range(1, 400):
#     train_loss = np.load('cpt/epoch_train{}.npy'.format(i))
#     test_loss = np.load('cpt/epoch_test{}.npy'.format(i))
#     train_losses.append(float(train_loss))
#     test_losses.append(float(test_loss))

mpl.rcParams['lines.linewidth'] = 3

import numpy as np

fig1 = plt.figure(figsize=(10, 6))
ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
ax1.set_title('Loss Plot', fontsize=16)
epoch = np.arange(1, len(train_losses)+1)
ax1.plot(epoch, train_losses, label="Train")
ax1.plot(epoch, test_losses, '--', label="Valid")

ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Value', fontsize=14)
plt.xticks(fontsize=12)
plt.gca().xaxis.set_major_locator(MaxNLocator(6))
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid()