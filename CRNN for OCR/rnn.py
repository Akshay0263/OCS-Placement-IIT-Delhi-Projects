"""
Assignment 5: CRNN For Text Recognition

Course Coordinator: Dr. Manojkumar Ramteke
Teaching Assistant: Abdur Rahman

This code is for educational purposes only. Unauthorized copying or distribution without the consent of the course coordinator is prohibited.
Copyright © 2024. All rights reserved.
"""
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        # To-do: Define the LSTM layer and a linear layer for required output size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # To-do: Implement the forward pass of the LSTM Module
        batch_size = input.size(0)
        # initial cell and hidden states
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(input.device)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(input.device)

        lstm_out, _ = self.lstm(input, (h0, c0))
        lstm_out = lstm_out[:,-1,:]
        output = self.linear(lstm_out)

        return output