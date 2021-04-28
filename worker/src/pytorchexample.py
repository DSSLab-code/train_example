# This PyTorch image classification example is based off
# https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/

from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
from train_model import LogisticRegression


model = LogisticRegression()
model.load_state_dict(torch.load("diagnoss2.pt"))
model.eval()

data0 = [[37.7, 0, 0, 1, 1, 0]]

# the label of data is zero
# for data1 the correct label is one, 1 is disease, 0 is not disease
data1 = [[41.5,0,1,1,0,1]]
input = Variable(torch.tensor(data1, dtype = torch.float32))
prediction = model(input).data.numpy()[:, 0]

# Print out the prediction of probability in precetage having the disease
with open("result.txt", "w") as outfile:
    outfile.write(str(prediction[0] * 100) + "%")
