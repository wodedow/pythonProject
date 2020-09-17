import torch


def print_hi(name):
    print(torch.cuda.is_available())
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


if __name__ == '__main__':
    print_hi('PyCharm')
