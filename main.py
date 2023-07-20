import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torch.nn as nn

# get ResNet18 pretrained model
model = models.resnet18(weights='DEFAULT')
# fix pretrained parameters
for param in model.parameters():
    param.requires_grad = False

# Resnet18 Normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean, std)])

# Get image data
orig_dataset = datasets.ImageFolder(root = './Image', transform=composed)

# Split image data randomly into train and validation set
size_train = int(0.9 * len(orig_dataset))
size_validation = len(orig_dataset) - size_train
train_dataset, validation_dataset = random_split(orig_dataset, [size_train, size_validation])

# Load datasets into DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=32)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=1)

# get number of classes of the dataset
num_class = len(orig_dataset.classes)
# get number of features of the dataset
num_features = model.fc.in_features 
# Replace the output layer model.fc of the neural network with a nn.Linear object
model.fc = nn.Linear(num_features, num_class)

# criterion
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

lr_scheduler=True
if lr_scheduler:
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01,step_size_up=5,mode="triangular2")

epochs=30
loss_list = []
accuracy_list = []
correct = 0
n_test = len(validation_dataset)

# Train and validate model
for epoch in range(epochs):
    loss_sublist = []
    for x, y in train_loader:
        model.train()
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss_sublist.append(loss.data.item())
        loss.backward()
        optimizer.step()

    loss_list.append(np.mean(loss_sublist))
    correct = 0
    
    for x_test, y_test in validation_loader:
        model.eval()
        z = model(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()

    accuracy = correct / n_test
    accuracy_list.append(accuracy)


def plot_stuff(COST,ACC):    
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(COST, color = color)
    ax1.set_xlabel('Iteration', color = color)
    ax1.set_ylabel('total loss', color = color)
    ax1.tick_params(axis = 'y', color = color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color = color)  # we already handled the x-label with ax1
    ax2.plot(ACC, color = color)
    ax2.tick_params(axis = 'y', color = color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.show()

plot_stuff(loss_list,accuracy_list)

torch.save(model.state_dict(), 'model.pt')

# print(len(orig_dataset))
# print(len(train_dataset))
# print(len(validation_dataset))
# print(len(train_loader))
# print(len(validation_loader))

# i=0
# zero = 0
# one = 0
# for image, label in train_dataset:
#     if label == 0:
#         zero += 1
#     else:
#         one +=1
#     print(label)
#     plt.imshow(image.permute(1, 2, 0).numpy())
#     plt.title('not_stop' if label == 0 else 'stop')
#     plt.show()
#     i+=1
#     if i==3:
#         break
# print(zero, one)