import torch


# yijuhx
def print_hi(name):
    print(torch.cuda.is_available())
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('PyCharm')
