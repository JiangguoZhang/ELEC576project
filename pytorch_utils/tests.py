'''
    Helper functions to be used in unit testing with pytest
'''
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

def is_learning(model,*batch):
    '''
        Asserts that all learnable layers in a model update during
        optimizer step.

        :param model (torch.nn.Module): Model to test
        :param batch (torch.Tensor): input to train model on, passed to the
             model as model(*batch)
    '''


    # A large value is needed here to be sure all paramters significantly
    # change after only 1 step
    optimizer = optim.SGD(model.parameters(),lr=0.1)

    #Collect the paramaters to train and their names
    original_learnables = []
    paramater_names = []
    for i in model.named_parameters():
        original_learnables.append(i[1].clone())
        paramater_names.append(i[0])

    ## Train Model
    forward_batch = []
    for b in batch:
        forward_batch.append(Variable(b))
    res = model(*forward_batch)

    if(isinstance(res, tuple)):
        #Assume that the first output of the tuple is what is trained on.
        res = res[0]

    loss = torch.pow(res,2).mean()
    loss.backward()
    optimizer.step()

    ## Get paramters after training
    modified_learnables = list(model.parameters())

    ## Test that paramters have changed
    for i in range(len(original_learnables)):
        assert (original_learnables[i] != modified_learnables[i]).data.any(), "Parameter " + paramater_names[i] + " not learning."
