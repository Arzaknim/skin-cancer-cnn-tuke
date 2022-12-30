import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from classes import classes


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


num_epochs = 15
batch_size = 4
learning_rate = 0.001

transform = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Resize((224, 224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

train_dataset = torchvision.datasets.ImageFolder(root="sorted_skin_cancer/train", transform=transform)

test_dataset = torchvision.datasets.ImageFolder(root="sorted_skin_cancer/test", transform=transform)

val_dataset = torchvision.datasets.ImageFolder(root="sorted_skin_cancer/val", transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                          shuffle=False)


# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# dataiter = iter(train_loader)
# images, labels = dataiter.__next__()
#
# # show images
# imshow(torchvision.utils.make_grid(images))

PATH = './cnn.pth'


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = F.relu(self.conv4(x))
        # print(x.shape)
        x = F.relu(self.conv5(x))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 256 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ConvNet().to(device)
# print(model.state_dict().values())
try:
    model.load_state_dict(torch.load(PATH))
    model.eval()
    model.train()
    # print(model.state_dict().values())
except Exception as e:
    print(f'Couldnt load the cnn.pth file, probably doesnt exist. {e}')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
epoch_acc = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 200 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images_val, labels_val in val_loader:
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)
            outputs = model(images_val)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels_val.size(0)
            n_correct += (predicted == labels_val).sum().item()

        acc = 100.0 * n_correct / n_samples
        epoch_acc.append(acc)
    model.train()

print('Finished Training')
torch.save(model.state_dict(), PATH)

model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(7)]
    n_class_samples = [0 for i in range(7)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            if i == len(labels):
                print("meow")
                break
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc:.4f} %')

    for i in range(7):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc:.4f} %')

    epochs = [x+1 for x in range(num_epochs)]
    plt.plot(epochs, epoch_acc)
    plt.xlabel("epochs")
    plt.ylabel("accuracy in percents")
    plt.show()
