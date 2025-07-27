import argparse
import gc
import pathlib
import time
from typing import List, Tuple, Union

import dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import u_net

from config import Config


def save_model_and_loss(
    output_dir: Union[pathlib.Path, str],
    model: torch.nn.Module,
    model_name: str,
    train_loss: List[float],
    valid_loss: List[float],
) -> None:
    """Saves the trained model & its loss function values in each training step.

    Args:
        output_dir: Absolute path to the output folder.
        model: Trained model.
        model_name: Name of the trained model.
        train_loss: Loss data generated during training.
        valid_loss: Loss data generated during validation.
    """
    output_dir = pathlib.Path(output_dir)
    if not output_dir.exists:
        pathlib.mkdir(output_dir)

    torch.save(model, output_dir / (model_name + ".pt"))
    torch.save(model.state_dict(), output_dir / (model_name + "_state_dict"))

    with open(output_dir / (model_name + "_training_loss.txt"), "w") as f:
        for element in train_loss:
            f.write(str(element) + "\n")
    with open(output_dir / (model_name + "_validation_loss.txt"), "w") as f:
        for element in valid_loss:
            f.write(str(element) + "\n")
    print("Model, state dictionary and loss values saved at: ", str(output_dir))


def train_seg_model(
    model: torch.nn.Module,
    device: torch.device,
    epochs: int,
    optimizer: object,
    loss_f: object,
    train_loader: torch.utils.data.dataloader.DataLoader,
    valid_loader: torch.utils.data.dataloader.DataLoader,
) -> Tuple[torch.nn.Module, List[float], List[float]]:
    """Performs training of the PointNet input model.

    Args:
        model: Model that is to be trained.
        device: PyTorch device (gpu or cpu) aka. hardware that will be used for computation.
        epochs: Number of training epochs.
        train_loader: PyTorch DataLoader object, iterator through training dataset.
        valid_loader: PyTorch DataLoader object, iterator through validation dataset.

    Returns:
        Trained model and loss function values.
    """
    gc.collect()  # Garbage colection.
    train_losses_all = []
    valid_losses_all = []
    time_start = time.time()
    print("Training the model!")

    for epoch in tqdm.trange(epochs):
        # Training loop.
        model.train()
        train_loss = 0.0
        for data in train_loader:
            image = data["image"].to(device).float()
            mask = data["mask"].to(device).float()
            # Clear the gradients.
            optimizer.zero_grad()
            # Forward pass.
            output = model(image)
            # Compute the loss.
            loss = loss_f(output, mask)
            # Backpropagate the loss & compute gradients.
            loss.backward()
            # Update weights.
            optimizer.step()

            train_loss += loss.item()

        train_loss_in_epoch = train_loss / len(train_loader)

        # Validation loop.
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():  # Turn off the gradients for validation.
            for data in valid_loader:
                image = data["image"].to(device).float()
                mask = data["mask"].to(device).float()
                # Forward pass.
                output = model(image)
                # Compute the loss.
                loss = loss_f(output, mask)

                valid_loss += loss.item()

        valid_loss_in_epoch = valid_loss / len(valid_loader)

        print("Epoch", epoch + 1, "complete!"),
        print("\tTraining Loss: ", round(train_loss_in_epoch, 4))
        print("\tValidation Loss: ", round(valid_loss_in_epoch, 4))
        train_losses_all.append(train_loss_in_epoch)
        valid_losses_all.append(valid_loss_in_epoch)

    time_end = time.time()
    print("Training finished!")
    print(f"Total training time: {int((time_end - time_start) / 60)} min.")

    return model, train_losses_all, valid_losses_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training and saving U-net or SegNet model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model that is to be trained.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Absolute path of the output directory.",
    )

    args = parser.parse_args()

    # Set the device & clean the memory
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    torch.cuda.empty_cache()

    # Load global config.
    config_file = Config()

    # Set the seeds:
    seed = config_file.config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get and set required training dataset parameters.
    data_dir = pathlib.Path(config_file.config["dataset"]["train"]["image"])
    mask_dir = pathlib.Path(config_file.config["dataset"]["train"]["mask"])
    # Get a list of names of the contents of the folder.
    all_images = list(path.name for path in pathlib.Path.iterdir(data_dir))

    train_image, eval_image = train_test_split(
        all_images, train_size=0.8, test_size=0.2
    )

    # Define transformations with random augmentations.
    train_transforms = transforms.Compose(
        [
            dataset.OneHotEncodeMask(dataset.COLOR_DICT),
            dataset.Resize(config_file.config["output_size"]),
            transforms.RandomApply(
                [dataset.ColorJitter(), dataset.GaussianBlur()], p=0.5
            ),
            dataset.ToTensor(),
        ]
    )

    # Define transformations with no random augmentations.
    # train_transforms=transforms.Compose([dataset.OneHotEncodeMask(dataset.COLOR_DICT), dataset.Resize(config_file.config["output_size"]), dataset.ToTensor()])

    train_set = dataset.SegMaskDataset(
        data_dir,
        mask_dir,
        train_image,
        eval_image,
        train=True,
        transforms=train_transforms,
    )
    valid_set = dataset.SegMaskDataset(
        data_dir,
        mask_dir,
        train_image,
        eval_image,
        train=False,
        transforms=train_transforms,
    )

    batch_size = config_file.config["batch_size"]

    print("Loading training and testing (validation) dataset!")
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Instantiate a model.
    model = u_net.UNet(3, 3, True)
    optimizer = optim.SGD(
        model.parameters(), momentum=0.99, lr=config_file.config["lr"]
    )
    # loss_f=nn.CrossEntropyLoss(reduction='mean')
    class_weights = torch.tensor([1.5, 1.0, 1.0], dtype=torch.float).to(device)
    loss_f = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    # loss_f=nn.BCELoss(weight=class_weights, reduction='mean')

    model.to(device)

    # Train the model.
    epochs = config_file.config["epochs"]
    trained_model, train_loss, valid_loss = train_seg_model(
        model=model,
        device=device,
        epochs=epochs,
        optimizer=optimizer,
        loss_f=loss_f,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )

    save_model_and_loss(
        args.output_path, model, args.model_name, train_loss, valid_loss
    )
