import random

import torch
import torchvision


def fetch_dataset():
    """ Collect MNIST """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )

    test_data = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )

    return train_data, test_data


def data_to_tensor(data):
    """ Loads dataset to memory, applies transform"""
    loader = torch.utils.data.DataLoader(data, batch_size=len(data))
    img, label = next(iter(loader))
    return img, label


def iid_partition_loader(data, bsz=10, n_clients=100):
    """ partition the dataset into a dataloader for each client, iid style
    """
    m = len(data)
    assert m % n_clients == 0
    m_per_client = m // n_clients
    assert m_per_client % bsz == 0

    client_data = torch.utils.data.random_split(
        data,
        [m_per_client for x in range(n_clients)]
    )
    client_loader = [
        torch.utils.data.DataLoader(x, batch_size=bsz, shuffle=True)
        for x in client_data
    ]
    return client_loader


def noniid_partition_loader(
    data, bsz=10, m_per_shard=300, n_shards_per_client=2
):
    """ semi-pathological client sample partition
    1. sort examples by label, form shards of size 300 by grouping points
       successively
    2. each client is 2 random shards
    most clients will have 2 digits, at most 4
    """

    # load data into memory
    img, label = data_to_tensor(data)

    # sort
    idx = torch.argsort(label)
    img = img[idx]
    label = label[idx]

    # split into n_shards of size m_per_shard
    m = len(data)
    assert m % m_per_shard == 0
    n_shards = m // m_per_shard
    shards_idx = [
        torch.arange(m_per_shard*i, m_per_shard*(i+1))
        for i in range(n_shards)
    ]
    random.shuffle(shards_idx)  # shuffle shards

    # pick shards to create a dataset for each client
    assert n_shards % n_shards_per_client == 0
    n_clients = n_shards // n_shards_per_client
    client_data = [
        torch.utils.data.TensorDataset(
            torch.cat([img[shards_idx[j]] for j in range(
                i*n_shards_per_client, (i+1)*n_shards_per_client)]),
            torch.cat([label[shards_idx[j]] for j in range(
                i*n_shards_per_client, (i+1)*n_shards_per_client)])
        )
        for i in range(n_clients)
    ]

    # make dataloaders
    client_loader = [
        torch.utils.data.DataLoader(x, batch_size=bsz, shuffle=True)
        for x in client_data
    ]
    return client_loader
