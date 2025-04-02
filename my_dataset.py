from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    # x 是一个n * 13 列表
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]