from typing import List, Any, Tuple
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from src.dataset import Image2ImageDataset


TrainDataset = Dataset
ValidDataset = Dataset


def _get_datasets(
    path_to_dataset,
    input_column_names: List[str],
    output_column_name: str,
    transform_train: Any,
    transform_valid: Any,
) -> Tuple[TrainDataset, ValidDataset]:
    dataset = Image2ImageDataset(
        path_to_dataset,
        input_column_names=input_column_names,
        output_column_name=output_column_name,
    )

    train_set = dataset
    valid_set = dataset

    train_set.transform = transform_train
    valid_set.transform = transform_valid

    train_proportion = 0.8
    valid_proportion = 0.1
    # test proportion is the remaining to 1

    train_size = int(train_proportion * len(train_set))
    valid_size = int(valid_proportion * len(train_set))

    indices = torch.randperm(
        len(train_set), generator=torch.Generator().manual_seed(42)
    )

    indices_train = indices[:train_size].tolist()
    indices_valid = indices[train_size : (train_size + valid_size)].tolist()

    train_set = torch.utils.data.Subset(train_set, indices_train)
    valid_set = torch.utils.data.Subset(valid_set, indices_valid)

    print("# of training data", len(train_set))
    print("# of validation data", len(valid_set))
    print("# of test data", len(dataset) - train_size - valid_size)

    return train_set, valid_set


TrainLoader = DataLoader
ValidLoader = DataLoader


def get_dataloaders(
    path_to_dataset,
    num_workers: int,
    batch_size: int,
    input_column_names: List[str],
    output_column_name: str,
    transform_train: Any,
    transform_valid: Any,
) -> Tuple[TrainLoader, ValidLoader]:

    train_set, valid_set = _get_datasets(
        path_to_dataset,
        input_column_names,
        output_column_name,
        transform_train,
        transform_valid,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, valid_loader
