import torch
from utils import CLI


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    CLI()


if __name__ == '__main__':
    main()
