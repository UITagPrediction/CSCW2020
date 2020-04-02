import numpy as np
import torch
import torch.utils.data

def data_generator(batch_sz=32):
    print("Start loading tags npy file....")
    x_train = np.load('../Data/npy-file/x_train.npy', allow_pickle = True)
    x_valid = np.load('../Data/npy-file/x_valid.npy', allow_pickle = True)
    x_test = np.load('../Data/npy-file/x_test.npy', allow_pickle = True)

    train_input = x_train[:,0]
    train_input = np.array([np.array(t).reshape(-1) for t in train_input])
    train_target = list(x_train[:,1:2].reshape(-1))
    print(train_input.shape)
    print(len(train_target))
    train = torch.Tensor(train_input).float()
    target = torch.Tensor(train_target).long()
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(train,target),
        batch_size=len(train_input), 
        shuffle=True,               
    )

    valid_input = x_valid[:,0]
    valid_input = np.array([np.array(t).reshape(-1) for t in valid_input])
    valid_target = list(x_valid[:,1:2].reshape(-1))
    valid_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(torch.Tensor(valid_input).float(),torch.Tensor(valid_target).long()),
        batch_size=len(valid_input), 
        shuffle=True,               
    )

    test_input = x_test[:,0]
    test_input = np.array([np.array(t).reshape(-1) for t in test_input])
    test_target = list(x_test[:,1:2].reshape(-1))
    test_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(torch.Tensor(test_input).float(),torch.Tensor(test_target).long()),
        batch_size=len(test_input), 
        shuffle=True,               
    )
    print("Done loading...")
    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    """ load data """
    train_loader, valid_loader, test_loader = data_generator()