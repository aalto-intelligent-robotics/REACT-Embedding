from typing import Tuple
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from react_embedding.dataset import InstancesDataset
from react_embedding.embedding_net import get_embedding_model
from react_embedding.losses import OnlineTripletLoss
from react_embedding.metrics import AverageNonzeroTripletsMetric
from react_embedding.trainer import fit
from react_embedding.triplet_selector import RandomNegativeTripletSelector
from react_embedding.utils import SquarePad


def form_ds(dataset_path: str) -> Tuple[InstancesDataset, InstancesDataset]:
    transform_val = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform_train = transforms.Compose( [
            SquarePad(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_ds = InstancesDataset(
        data_root=f"{dataset_path}/train", transforms=transform_train
    )
    test_ds = InstancesDataset(
        data_root=f"{dataset_path}/val", transforms=transform_val
    )
    return train_ds, test_ds


def main(args: argparse.Namespace):
    train_ds, test_ds = form_ds(dataset_path=args.dataset_path)
    trainloader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    testloader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=8
    )
    model = get_embedding_model(backbone="efficientnet_b2")
    margin = 1.0
    loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
    lr = args.lr
    weight_decay=1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 50
    train_hist, val_hist = fit(
        train_loader=trainloader,
        val_loader=testloader,
        model=model.backbone.cuda(),
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=n_epochs,
        cuda=True,
        log_interval=log_interval,
        metrics=[AverageNonzeroTripletsMetric()],
        start_epoch=0,
    )
    plt.plot(train_hist)
    plt.plot(val_hist)
    plt.show()
    torch.save(model.state_dict(), args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset-path", type=str, default="./dataset_coffee_room_1/"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-o", "--output", type=str, default="embedding_coffee_room.pth")
    args = parser.parse_args()
    main(args)
