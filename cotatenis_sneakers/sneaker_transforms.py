from torchvision import transforms


class PadToSize:
    """
    Permet de rajouter du blanc sur les images pour les mettre à la taille souhaitée.
    """

    def __init__(self, size, fill=255):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        width, height = img.size
        pad_width = max(0, self.size[0] - width)
        pad_height = max(0, self.size[1] - height)

        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return transforms.functional.pad(img, padding, fill=self.fill)


class UnNormalize(object):
    """
    Permet de reprinter les images après les avoir normalisées (en les dénormalisant)
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def get_transform():
    transform = transforms.Compose(
        [
            PadToSize((400, 400)),
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform