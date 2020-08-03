import torch
from torch.utils import data
import json
import random
from torchvision import transforms
import cv2
import numpy as np
import lmdb


def get_alphabet(f_alphabet):
    with open(f_alphabet, 'r', encoding='utf-8') as f:
        alphabet = sorted(json.load(f))
    return alphabet


class AlignCollate(object):
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, img_label):
        img_label.sort(key=lambda x: x[2], reverse=True)
        imgs, labels, lengths, index = zip(*img_label)
        imgs = torch.stack(imgs, dim=0)
        lengths_tensor = torch.LongTensor(lengths)
        # label_tensor = torch.LongTensor(list(itertools.zip_longest(*labels, fillvalue=padding)))
        label_tensor = torch.nn.utils.rnn.pad_sequence(labels, padding_value=self.padding)
        return [imgs, (label_tensor, lengths_tensor), index]


class LmdbDataset(data.Dataset):
    def __init__(self, lmdb_path, input_size, transform, alphabet, is_train=True):
        super(LmdbDataset, self).__init__()
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.is_train = is_train
        self.input_size = input_size[::-1]  # PIL resize [w, h]
        self.alphabet = alphabet

        self.env = lmdb.open(lmdb_path, max_readers=32, readonly=True)
        assert self.env is not None, "cannot create lmdb obj from %s" % lmdb_path
        self.txn = self.env.begin()
        self.count = int(self.txn.get('nSamples'.encode()).decode())
        self.nSub = []
        for i in range(4, 15):
            key_sub = 'num_' + str(i)
            self.nSub.append(int(self.txn.get(key_sub.encode()).decode()))
        print('Loaded {0} {1} images'.format(self.count, 'training' if is_train else 'test'))
    
    @property
    def pad_token(self):
        return 1

    def convert2id(self, s):
        ids = [self.alphabet.index(si) for si in s]
        return torch.LongTensor(ids)

    def __getitem__(self, idx):
        image_key = '%d_0_image' % idx
        try:
            img_binary = self.txn.get(image_key.encode())
            imgbuf = np.frombuffer(img_binary, dtype=np.uint8)
            image = cv2.imdecode(imgbuf, 1)
        except Exception as e:
            print("Error {0} at image {1}".format(e, image_key))
            return None
        if self.is_train:
            image = random_augmentation(image)
        image = cv2.resize(image, tuple(self.input_size))
        if self.transform:
            image = self.transform(image)
        
        label_key = "%d_0_label" % idx
        label = self.txn.get(label_key.encode()).decode()
        label = self.convert2id(label)
        label = label + 2
        label = torch.cat((label, torch.tensor([1])))  # end up with EoS(1)
        length = len(label)
        return image, label, length, idx
    
    def __len__(self):
        return self.count


class DoubleShuffleSampler(data.sampler.Sampler):
    """
    A sampler which could be used to dataset where the sequence data has several groups with different length.
    Dataset is like this: [...n1...], [...n2...], [...n3...], ...
    where each group has their size different from others.
    We want a single batch to be from a single group, while keeping the batches shuffled.
    So we just follow the 2 steps: randomly shuffle indexes with each group; shuffle all batches.
    """
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        all_indexes = []    # index of the items in dataset
        all_indexes2 = []   # index of the above index value
        start_index = 0
        for i, n_sub in enumerate(self.data_source.nSub):
            sub_indexes = list(range(start_index, start_index + n_sub))
            random.shuffle(sub_indexes)     # the first shuffle within group
            all_indexes.extend(sub_indexes)
            num_sub_batches = n_sub // self.batch_size
            num_remain = n_sub % self.batch_size
            sub_indexes2 = start_index + np.arange(num_sub_batches) * self.batch_size
            sub_indexes2 = sub_indexes2.tolist()
            if num_remain != 0:
                sub_indexes2.append(start_index + n_sub - self.batch_size)  # store the start index of all batches
            all_indexes2.extend(sub_indexes2)
            start_index += n_sub
        random.shuffle(all_indexes2)    # the second shuffle about batches(denoted by the start index)
        # complete other indexes besides the start ones. Use numpy to accelerate
        tmp_all_indexes2 = np.array(all_indexes2)
        tmp_expand = tmp_all_indexes2.repeat(self.batch_size).reshape(len(all_indexes2), self.batch_size)
        tmp_other = np.arange(self.batch_size).reshape(1, -1)
        tmp = tmp_expand + tmp_other
        tmp = tmp.ravel().tolist()
        all_indexes = np.array(all_indexes)
        final_indexes = all_indexes[tmp]
        return iter(final_indexes.tolist())


def random_augmentation(image, allow_crop=True):
    f = ImageTransfer(image)
    seed = random.randint(0, 6)     # 0: original image used
    switcher = random.random() if allow_crop else 1.0
    if seed == 1:
        image = f.add_noise()
    elif seed == 2:
        image = f.change_contrast()
    elif seed == 3:
        image = f.change_hsv()
    elif seed == 4:
        a = random.random() * 0.4 + 0.8
        gamma = random.random() * 2.2
        image = f.gamma_transform(a=a, gamma=gamma)
    elif seed >= 5:
        f1 = ImageTransfer(f.add_noise())
        f2 = ImageTransfer(f1.change_hsv())
        f3 = ImageTransfer(f2.gamma_transform(1.0, 1.5))
        image = f3.change_contrast()
    if switcher < 0.4:
        fn = ImageTransfer(image)
        image = fn.slight_crop()
    elif switcher < 0.7:
        fn = ImageTransfer(image)
        image = fn.perspective_transform()
    return image


class ImageTransfer(object):
    """crop, add noise, change contrast, color jittering"""
    def __init__(self, image):
        """image: a ndarray with size [h, w, 3]"""
        self.image = image

    def slight_crop(self):
        h, w = self.image.shape[:2]
        k = random.random() * 0.08  # 0.0 <= k <= 0.1
        ch, cw = int(h * 0.9), int(w - k * h)     # cropped h and w
        hs = random.randint(0, h - ch)      # started loc
        ws = random.randint(0, w - cw)
        return self.image[hs:hs+ch, ws:ws+cw]

    def add_noise(self):
        img = self.image * (np.random.rand(*self.image.shape) * 0.2 + 0.8)
        img = img.astype(np.uint8)
        return img

    def change_contrast(self):
        if random.random() < 0.5:
            k = random.randint(7, 9) / 10.0
        else:
            k = random.randint(11, 13) / 10.0
        b = 128 * (k - 1)
        img = self.image.astype(np.float)
        img = k * img - b
        img = np.maximum(img, 0)
        img = np.minimum(img, 255)
        img = img.astype(np.uint8)
        return img

    def perspective_transform(self):
        h, w = self.image.shape[:2]
        short = min(h, w)
        gate = int(short * 0.05)
        mrg = []
        for _ in range(8):
            mrg.append(random.randint(0, gate))
        pts1 = np.float32(
            [[mrg[0], mrg[1]], [w - 1 - mrg[2], mrg[3]], [mrg[4], h - 1 - mrg[5]], [w - 1 - mrg[6], h - 1 - mrg[7]]])
        pts2 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(self.image, M, (w, h))

    def gamma_transform(self, a=1.0, gamma=2.0):
        image = self.image.astype(np.float)
        image = image / 255
        image = a * (image ** gamma)
        image = image * 255
        image = np.minimum(image, 255)
        image = image.astype(np.uint8)
        return image

    def change_hsv(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        s = random.random()
        def ch_h():
            dh = random.randint(2, 11) * random.randrange(-1, 2, 2)
            img[:, :, 0] = (img[:, :, 0] + dh) % 180
        def ch_s():
            ds = random.random() * 0.25 + 0.7
            img[:, :, 1] = ds * img[:, :, 1]
        def ch_v():
            dv = random.random() * 0.35 + 0.6
            img[:, :, 2] = dv * img[:, :, 2]
        if s < 0.25:
            ch_h()
        elif s < 0.50:
            ch_s()
        elif s < 0.75:
            ch_v()
        else:
            ch_h()
            ch_s()
            ch_v()
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def get_dataloader(lmdb_dir, alphabet, input_size, trsf, batch_size, is_train=True):
    kwargs = {'num_workers': 4, 'pin_memory': True}
    dataset = LmdbDataset(lmdb_dir, input_size, trsf, alphabet, is_train)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, sampler=DoubleShuffleSampler(dataset, batch_size),
                                 collate_fn=AlignCollate(dataset.pad_token), **kwargs)
    return dataloader


if __name__ == '__main__':
    lmdb_dir = "/home/dataset/TR/synth_cn/train_lmdb/"
    f_alphabet = "/home/dataset/TR/synth_cn/alphabet.json"
    alphabet = get_alphabet(f_alphabet)
    train_trsf = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    myloader = get_dataloader(lmdb_dir, alphabet, [32, 256], train_trsf, 8)
    # display some results
    print("Num_batches: %d" % len(myloader))
    dataiter = iter(myloader)
    images, (labels, lengths), indexes = dataiter.next()
    print("BatchSize: {0}, ImageSize: {1}".format(images.size(0), images[2].size()))
    print("image indexes are: {}".format(indexes))
    import torchvision as tv
    images = tv.utils.make_grid(images)
    images = (images * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) +
              torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) * 255
    cv2.imwrite("batch.jpg", images.numpy().transpose(1, 2, 0))
    print(labels)
    print("Note: batch display saved in file batch.jpg")
    print('The first gt is -- ' + ''.join('%s' % alphabet[labels[:, 0][j].item() - 2] for j in range(
        len(labels[:, 0]) - 1)))
