import torch

device      = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor  = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
