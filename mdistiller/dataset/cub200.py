import os
import tarfile

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url, check_integrity
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

__all__ = ['CUB2011Metric']


class CUB2011(ImageFolder):
    image_folder = 'CUB_200_2011/images'
    base_folder = 'CUB_200_2011/'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    checklist = [
        ['001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg', '4c84da568f89519f84640c54b7fba7c2'],
        ['002.Laysan_Albatross/Laysan_Albatross_0001_545.jpg', 'e7db63424d0e384dba02aacaf298cdc0'],
        ['198.Rock_Wren/Rock_Wren_0001_189289.jpg', '487d082f1fbd58faa7b08aa5ede3cc00'],
        ['200.Common_Yellowthroat/Common_Yellowthroat_0003_190521.jpg', '96fd60ce4b4805e64368efc32bf5c6fe']
    ]

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = root
        if download:
            download_url(self.url, root, self.filename, self.tgz_md5)

            if not self._check_integrity():
                cwd = os.getcwd()
                tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
                os.chdir(root)
                tar.extractall()
                tar.close()
                os.chdir(cwd)

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        super(CUB2011, self).__init__(os.path.join(root, self.image_folder),
                                      transform=transform,
                                      target_transform=target_transform)

    def _check_integrity(self):
        for f, md5 in self.checklist:
            fpath = os.path.join(self.root, self.image_folder, f)
            if not check_integrity(fpath, md5):
                return False
        return True


class CUB2011Classification(CUB2011):
    def __init__(self, root, train=False, transform=None, target_transform=None, download=False):
        CUB2011.__init__(self, root, transform=transform, target_transform=target_transform, download=download)

        with open(os.path.join(root, self.base_folder, 'images.txt'), 'r') as f:
            id_to_image = [l.split(' ')[1].strip() for l in f.readlines()]

        with open(os.path.join(root, self.base_folder, 'train_test_split.txt'), 'r') as f:
            id_to_istrain = [int(l.split(' ')[1]) == 1 for l in f.readlines()]

        train_list = [os.path.join(root, self.image_folder, id_to_image[idx]) for idx in range(len(id_to_image)) if
                      id_to_istrain[idx]]
        test_list = [os.path.join(root, self.image_folder, id_to_image[idx]) for idx in range(len(id_to_image)) if
                     not id_to_istrain[idx]]

        if train:
            self.samples = [(img_file_pth, cls_ind) for img_file_pth, cls_ind in self.imgs
                            if img_file_pth in train_list]
        else:
            self.samples = [(img_file_pth, cls_ind) for img_file_pth, cls_ind in self.imgs
                            if img_file_pth in test_list]
        self.imgs = self.samples


class CUB2011Classification_Instance(CUB2011):
    def __init__(self, root, train=False, transform=None, target_transform=None, download=False):
        CUB2011.__init__(self, root, transform=transform, target_transform=target_transform, download=download)

        with open(os.path.join(root, self.base_folder, 'images.txt'), 'r') as f:
            id_to_image = [l.split(' ')[1].strip() for l in f.readlines()]

        with open(os.path.join(root, self.base_folder, 'train_test_split.txt'), 'r') as f:
            id_to_istrain = [int(l.split(' ')[1]) == 1 for l in f.readlines()]

        train_list = [os.path.join(root, self.image_folder, id_to_image[idx]) for idx in range(len(id_to_image)) if
                      id_to_istrain[idx]]
        test_list = [os.path.join(root, self.image_folder, id_to_image[idx]) for idx in range(len(id_to_image)) if
                     not id_to_istrain[idx]]

        if train:
            self.samples = [(img_file_pth, cls_ind) for img_file_pth, cls_ind in self.imgs
                            if img_file_pth in train_list]
        else:
            self.samples = [(img_file_pth, cls_ind) for img_file_pth, cls_ind in self.imgs
                            if img_file_pth in test_list]
        self.imgs = self.samples
        self.train = train

    def __getitem__(self, index):

        (img_file_pth, cls_ind) = self.imgs[index]

        img=Image.open(img_file_pth).convert('RGB')
        # print(img.size)
        target=cls_ind

        if self.transform is not None:
            img_teacher = self.transform(img)
            img_student = img_teacher

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_teacher, target, index

class CUB2011Metric(CUB2011):
    num_training_classes = 100

    def __init__(self, root, train=False, split='none', transform=None, target_transform=None, download=False):
        CUB2011.__init__(self, root, transform=transform, target_transform=target_transform, download=download)

        if train:
            if split == 'train':
                self.classes = self.classes[:(self.num_training_classes - 20)]
            elif split == 'val':
                self.classes = self.classes[(self.num_training_classes - 20):self.num_training_classes]
            else:
                self.classes = self.classes[:self.num_training_classes]
        else:
            self.classes = self.classes[self.num_training_classes:]

        self.class_to_idx = {cls_name: cls_ind for cls_name, cls_ind in self.class_to_idx.items()
                             if cls_name in self.classes}
        self.samples = [(img_file_pth, cls_ind) for img_file_pth, cls_ind in self.imgs
                        if cls_ind in self.class_to_idx.values()]
        self.imgs = self.samples


def get_cub200_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    """
    cifar 100
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    if is_instance:
        train_set = CUB2011Classification_Instance(root='./data',
                                          download=True,
                                          train=True,
                                          transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = CUB2011Classification(root='./data', train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=True)

    test_set = CUB2011Classification(root='./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size / 2),
                             shuffle=False,
                             num_workers=int(num_workers / 2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader
