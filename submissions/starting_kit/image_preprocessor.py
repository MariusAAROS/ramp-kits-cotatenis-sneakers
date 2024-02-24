from cotatenis_sneakers.sneaker_transforms import get_transform

def transform(x):
    transform = get_transform()
    x = transform(x)
    return x