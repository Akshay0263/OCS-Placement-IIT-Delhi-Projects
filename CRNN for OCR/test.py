"""
Assignment 5: CRNN For Text Recognition

Course Coordinator: Dr. Manojkumar Ramteke
Teaching Assistant: Abdur Rahman

This code is for educational purposes only. Unauthorized copying or distribution without the consent of the course coordinator is prohibited.
Copyright © 2024. All rights reserved.
"""

import torch
import argparse
from tqdm import tqdm
import torch.utils.data
from model import Model
from utils import ConverterForCTC
from dataset import AlignCollate, Num10kDataset

def validation(model, criterion, evaluation_loader, converter, args, device):
    """ Evaluation Function """
    # To-do: Write the code to evaluate the model
        # Iterate over the data and encode the labels
        # Forward pass the image through the model
        # Calculate the loss using the criterion
        # Decode the model predictions using the converter
        # Return the average loss and accuracy

def test(args, device):
    args.device = device
    print('\n'+'-' * 80)
    print('Device : {}'.format(device))
    print('-' * 80 + '\n')
    
    # Load the Validation Dataset
    AlignCollate_valid = AlignCollate(imgH=args.imgH, imgW=args.imgW)
    valid_dataset = Num10kDataset(args.valid_data)
    print("Loaded Validation Dataset, Length: ", len(valid_dataset))
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size,
        shuffle=True,
        collate_fn=AlignCollate_valid, pin_memory=False)
    
    # To-do: Similar to train.py, write the code below
        # Create an instance of the ConverterForCTC
        # Load the Model from the saved_model path
        # Call validation function to evaluate the model
        # Print the accuracy and loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=0, help='Maximum label length') # DECIDE APPROPRIATELY
    parser.add_argument('--imgH', type=int, default=0, help='the height of the input image') # DECIDE APPROPRIATELY
    parser.add_argument('--imgW', type=int, default=0, help='the width of the input image') # DECIDE APPROPRIATELY
    
    """ Model Architecture - DO NOT CHANGE """
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel for CNN')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel for CNN')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    
    args = parser.parse_args()
    
    """ vocab / character number configuration """
    args.character = "0123456789"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)
    
    test(args, device)