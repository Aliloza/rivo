"""Augmentation pipelines for training and evaluation."""

from torchvision import transforms


def get_train_transforms(input_size: int = 224) -> transforms.Compose:
    """Return augmentation + normalisation pipeline for training.

    Augmentations applied:
    - Resize to ``input_size``
    - Random horizontal flip
    - Random rotation (±15 °)
    - Random contrast / brightness adjustment
    - Gaussian blur (simulates optical noise)
    - Convert to tensor
    - ImageNet normalisation

    Args:
        input_size (int): Target image size (height = width).

    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_eval_transforms(input_size: int = 224) -> transforms.Compose:
    """Return resize + normalisation pipeline for validation / test.

    No stochastic augmentations are applied.

    Args:
        input_size (int): Target image size (height = width).

    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
