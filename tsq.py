import torch    
use_gpu = torch.cuda.is_available()
print(torch.cuda.is_available())
print("use_gpu is: ",use_gpu)
