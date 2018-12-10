from __future__ import print_function, division
import numpy as np
import torch
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import torch.nn as nn
import urllib
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from matplotlib import cm
import torchvision
import torchvision.transforms as transforms


# Loading input images and targets
url_response = urllib.urlretrieve("https://storage.googleapis.com/ml2-group4-project/all_images.npy", "all_images.npy")
x = np.load("all_images.npy")

url_response = urllib.urlretrieve("https://storage.googleapis.com/ml2-group4-project/all_labels.npy", "all_labels.npy")
y = np.load("all_labels.npy")

# Decoding hot-encoded labels
y = [np.where(r==1)[0][0] for r in y]
y = np.asarray(y)

# Split x and y
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()

# input_size = 3 x 32 x 32
input_size = 3072
hidden_size = 500
num_classes = 5
num_epochs = 100
batch_size = 20
learning_rate = .0001

train = data_utils.TensorDataset(X_train, y_train)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
test = data_utils.TensorDataset(X_test, y_test)
test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

classes = ('drink', 'food', 'inside', 'outside', 'menu')

def imshow(img):
    #img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), aspect='auto', norm=cm.colors.Normalize(vmax=1, vmin=0 ))
    #plt.imshow(npimg)
    plt.show()


dataiter = iter(train_loader)
images, labels = dataiter.next()


#imshow(torchvision.utils.make_grid(images.cpu()))
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sm = nn.Softmax()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sm(out)
        out = self.fc2(out)
        return out


net = Net(input_size, hidden_size, num_classes)

get_cuda = True

if torch.cuda.is_available() and get_cuda:
    net = net.cuda()

#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(num_epochs):

    for i, data in enumerate(train_loader):

        images, labels = data
        if torch.cuda.is_available() and get_cuda:
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        images = images.view(-1, 3 * 32 * 32)
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #print (loss.data[0])

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                % (epoch + 1, num_epochs, i + 1, len(train) // batch_size, loss.data[0]))

dataiter = iter(test_loader)
images, labels = dataiter.next()


correct = 0
total = 0
for images, labels in test_loader:

    images = Variable(images.view(-1, 3 * 32 * 32))
    if torch.cuda.is_available() and get_cuda:
        images = images.cuda()

    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    if torch.cuda.is_available() and get_cuda:
        correct += (predicted == labels.cuda()).sum()
    else:
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 100000 test images: %d %%' % (100 * correct / total))

imshow(torchvision.utils.make_grid(images.cpu(), normalize=True, scale_each=True))
print('Actual: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
outputs = net(images)

_, predicted = torch.max(outputs.data, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in test_loader:
    images, labels = data
    images = Variable(images.view(-1, 3 * 32 * 32))

    if torch.cuda.is_available() and get_cuda:
        images = images.cuda()
        labels = labels.cuda()

    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)

    c = (predicted == labels).squeeze()

    for i in range(batch_size):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1

for i in range(5):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
