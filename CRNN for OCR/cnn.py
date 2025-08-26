"""
Assignment 5: CRNN For Text Recognition

Course Coordinator: Dr. Manojkumar Ramteke
Teaching Assistant: Abdur Rahman

This code is for educational purposes only. Unauthorized copying or distribution without the consent of the course coordinator is prohibited.
Copyright © 2024. All rights reserved.
"""

import torch.nn as nn
import torchvision.models as models

class CNNModule(nn.Module):
    """ The CNN Model for feature extraction """
    def __init__(self, input_channel=1, output_channel=512):
        super(CNNModule, self).__init__()
        # To-do: Define the layers for the CNN Module
        vgg16 = models.vgg16(pretrained=True)

        for param in vgg16.parameters():
            param.requires_grad = False
        
        self.features = vgg16.features
        self.classifier = nn.Sequential(
            nn.Linear(512*160, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, output_channel)
        )

    def forward(self, input):
        # To-do: Implement the forward pass of the CNN Moduler
        x = self.features(input) # forward pass to feature extractor
        print("Shape after CNN features:", x.shape)
        x = x.view(x.size(0), -1) # flatten
        print("Shape after flattening in CNNModule:", x.shape)
        x = self.classifier(x) # custom classifier
        print("Shape after classifier:", x.shape)
        return x