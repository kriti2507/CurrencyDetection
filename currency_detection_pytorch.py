from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import time
# from imageio import imread
import pandas as pd
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn

train_csv_file = "./training_labels.csv"
val_csv_file = "./validation_labels.csv"

# train_data_name = pd.read_csv(train_csv_file)
# val_data_name = pd.read_csv(val_csv_file)

train_data_dir = "./training_data_pytorch/training_data_pytorch/"
val_data_dir = "./validation_data_pytorch/validation_data_pytorch/"

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(), transforms.Normalize(mean, std)]) 
	
class Dataset(Dataset):

	#Constructor
	def __init__(self, csv_file, data_dir, transform = None):

		self.data_dir = data_dir
		self.transform = transform
		self.data_name = pd.read_csv(csv_file)
		self.len = self.data_name.shape[0]

	# Get the length
	def __len__(self):
		return self.len

	#Getter
	def __getitem__(self, idx):

		img_name = self.data_dir + self.data_name.iloc[idx, 2]
		image = Image.open(img_name)
		y = self.data_name.iloc[idx,3]

		if self.transform:
			image = self.transform(image)

		return image, y

train_dataset = Dataset(transform = composed, csv_file = train_csv_file, data_dir = train_data_dir)
val_dataset = Dataset(transform = composed, csv_file = val_csv_file, data_dir = val_data_dir)

#Load the pre-trained model resnet18
model = models.resnet18(pretrained = True)

#Set the parameter cannot be trained for the pre-trained model
for param in model.parameters():
	param.requires_grad = False

#Re-defined the last layer
model.fc = nn.Linear(512,7)

#Loss function
criterion = nn.CrossEntropyLoss()

#Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=15, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=10, shuffle=True)

#Adam optimizer
optimizer = torch.optim.Adam([parameters for parameters in model.parameters() if parameters.requires_grad], lr = 0.003)

#Train
N_EPOCHS = 20
loss_list = []
accuracy_list = []
correct = 0
n_test = float(len(val_dataset))

for epoch in range(N_EPOCHS):
	loss_sublist = []
	for x,y in train_loader:
		model.train()
		optimizer.zero_grad()
		z = model(x)
		loss = criterion(z,y)
		loss_sublist.append(loss.data.item())
		loss.backward()
		optimizer.step()
	loss_list.append(np.mean(loss_sublist))
	
	correct = 0
	for x_test, y_test in val_loader:
		print(x_test.shape)
		print(y_test.shape)
		model.eval()
		z = model(x_test)
		_,yhat = torch.max(z.data,1)
		correct += (yhat == y_test).sum().item()

	accuracy = correct/n_test
	print(accuracy)
	accuracy_list.append(accuracy)

torch.save(model.state_dict(), "")

x = np.arange(len(loss_list))
plt.plot(x,loss_list)
plt.title('Average Loss per Epoch vs epoch')
plt.xlabel('Epoch')
plt.ylabel('Average loss per epoch')
plt.show()

x = np.arange(N_EPOCHS)
plt.plot(x,accuracy_list)
plt.title('Accuracy per Epoch vs epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

#Testing
look_up = {0: 'predicted: $5'
		   , 1: 'predicted: $10'
		   , 2: 'predicted: $20'
		   , 3: 'predicted: $50'
		   , 4: 'predicted: $100'
		   , 5: 'predicted $200'
		   , 6: 'predicted $500'}
random.seed(0)
numbers = random.sample(range(70), 5)

# Type your code here

# def plot_random_image(numbers):
model.eval()
count = 1
for i in numbers:
	img , y = val_dataset.__getitem__(idx = i)
	print("Image " + str(count))
	val_image_name = val_data_dir + str(i) + ".jpeg"
	image = Image.open(val_image_name)
	plt.imshow(image)
	plt.show()
	print(look_up[y])
	shape = list(img.size())
	img = img.view(1, shape[0], shape[1], shape[2])
	z = model(img)
	_,yhat = torch.max(z.data,1)
	count+= 1
	if yhat.item() == y:
		print(" **************Correctly classified************* ")
	else:
		print("**************Mis classified**************")