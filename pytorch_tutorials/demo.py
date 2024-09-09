import torch
import torch.nn as nn

import pprint
pp = pprint.PrettyPrinter()

list_of_lists = [
    [1, 2, 3],
    [4, 5, 6]
]
print(list_of_lists)

# initializing a pytorch tensor
data = torch.tensor([
    [0, 1],
    [2, 3],
    [4, 5]
])
print(data)