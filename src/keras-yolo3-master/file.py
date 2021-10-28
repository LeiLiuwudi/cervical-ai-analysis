import os


def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f


def main():
    base = 'E:\\graduationProject\\VGG16_TF-master\\data\\dataset\\all'
    for i in findAllFile(base):
        print(i)


if __name__ == '__main__':
    for i in range(10):
        print(0)
