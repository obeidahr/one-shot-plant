import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


import streamlit as st # type: ignore

# Creating some helper functions
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                #Look untill the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                #Look untill a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

#create the Siamese Neural Network
class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256,2)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

# Resize the images and transform to tensors
transformation = transforms.Compose([transforms.Resize((100,100)),
                                     transforms.ToTensor()
                                    ])
# Load the saved model
net = SiameseNetwork()
net.load_state_dict(torch.load('one_shot_model.h5',map_location=torch.device('cpu')))
net.eval()

# # Locate the test dataset and load it into the SiameseNetworkDataset
# folder_dataset_test = datasets.ImageFolder(root="data/plants/testing")
# siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
#                                         transform=transformation)
# test_dataloader = DataLoader(siamese_dataset, num_workers=2, batch_size=1, shuffle=True)

# # Grab one image that we are going to test
# dataiter = iter(test_dataloader)
# #x0, _, _ = next(dataiter)
# for i in range(20):
#     # Iterate over 10 images and test them with the first image (x0)
#     x0, x1, label2 = next(dataiter)
#     # Concatenate the two images together
#     concatenated = torch.cat((x0, x1), 0)

#     output1, output2 = net(x0, x1)
#     euclidean_distance = F.pairwise_distance(output1, output2)
#     if euclidean_distance < 0.5:
#         imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')

# Streamlit App
st.title('One Shot Learning :coffee:')

st.title("Similar Image with Support Images")

uploaded_image = st.file_uploader("Upload your Query Image",type=["jpg", "jpeg", "png"], accept_multiple_files=False)
#uploaded_image_title = st.text_input("Title for Uploaded Image")

support_images = st.file_uploader("Upload Multiple Support Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
# support_images_titles = []
# for i, support_image in enumerate(support_images):
#     title = st.text_input(f"Title for Support Image {i+1}")
#     support_images_titles.append(title)

sim_images = []
if uploaded_image is not None and support_images is not None:
    # Process the uploaded image
    image1 = Image.open(uploaded_image)
    processed_image = transformation(image1).unsqueeze(0)  # Add a batch dimension

    for i, support_image in enumerate(support_images):
        support_image = Image.open(support_image)
        processed_support_image = transformation(support_image).unsqueeze(0)

        # Apply the transformation to the images and classify
        output1, output2 = net(processed_image, processed_support_image)
        euclidean_distance = F.pairwise_distance(output1, output2)

        if euclidean_distance < 0.8:
            sim_images.append((support_image))

    st.success(f'Predicted similar images: {len(sim_images)}')

    for i, image in enumerate(sim_images):
        st.image(image, use_column_width=True)