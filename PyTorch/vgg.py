import torch
from torch import optim, nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import time

DOWNLOAD_CIFAR = True
batch_size = 32
lr = 0.01
step_size = 10
epoch_num = 50
num_print = int(50000 // batch_size // 4)

train_data = torchvision.datasets.CIFAR10(
	root='.',
	train=True,
	transform=torchvision.transforms.ToTensor(),
	download=DOWNLOAD_CIFAR
)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = torchvision.datasets.CIFAR10(
	root='.',
	train=False,
	transform=torchvision.transforms.ToTensor(),
	download=DOWNLOAD_CIFAR
)
test_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def image_show(img):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()


def label_show(loader):
	global classes
	dataiter = iter(loader)
	images, labels = dataiter.__next__()
	image_show(make_grid(images))
	print(''.join('%5s' % classes[labels[j]] for j in range(batch_size)))
	return images, labels


label_show(train_loader)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Vgg16Net(nn.Module):
	def __init__(self):
		super(Vgg16Net, self).__init__()

		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 128, 3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.layer3 = nn.Sequential(
			nn.Conv2d(128, 256, 3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.layer4 = nn.Sequential(
			nn.Conv2d(256, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.layer5 = nn.Sequential(
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=512),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.conv_layer = nn.Sequential(
			self.layer1,
			self.layer2,
			self.layer3,
			self.layer4,
			self.layer5
		)

		self.fc = nn.Sequential(
			nn.Linear(512, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),

			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),

			nn.Linear(4096, 1000)
		)

	def forward(self, x):
		x = self.conv_layer(x)
		x = x.view(-1, 512)
		x = self.fc(x)
		return x


model = Vgg16Net().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5, last_epoch=-1)

loss_list = []
start = time.time()

for epoch in range(epoch_num):
	running_loss = 0.0
	for i, (inputs, labels) in enumerate(train_loader, 0):
		inputs, labels = inputs.to(device), labels.to(device)

		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels).to(device)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		loss_list.append(loss.item())
		if i % num_print == num_print - 1:
			print('[%d epoch, %d] loss: %.6f' % (epoch + 1, i + 1, running_loss / num_print))
			running_loss = 0.0
	lr_1 = optimizer.param_groups[0]['lr']
	print('learn_rate : %.15f' % lr_1)
	scheduler.step()

end = time.time()
print(f'time:{end - start}')

torch.save(model, './model.pkl')
model = torch.load('./model.pkl')

model.eval()
correct = 0.0
total = 0

with torch.no_grad():
	for inputs, labels in test_loader:
		inputs, labels = inputs.to(device), labels.to(device)
		outputs = model(inputs)
		pred = outputs.argmax(dim=1)
		total += inputs.size(0)
		correct += torch.eq(pred, labels).sum().item()
print("Accuary of the network on the 10000 test images: %.2f %%" % (100.0 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for inputs, labels in test_loader:
	inputs, labels = inputs.to(device), labels.to(device)
	outputs = model(inputs)
	pred = outputs.argmax(dim=1)
	c = (pred == labels.to(device)).squeeze()
	for i in range(4):
		label = labels[i]
		class_correct[label] += float(c[i])
		class_total[label] += 1

	for i in range(10):
		print('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
