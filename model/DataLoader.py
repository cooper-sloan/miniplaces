import os
import numpy as np
import scipy.misc
import h5py
import cv2
np.random.seed(123)

# loading data from .h5
class DataLoaderH5(object):
    def __init__(self, **kwargs):
        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']

        # read data info from lists
        f = h5py.File(kwargs['data_h5'], "r")
        self.im_set = np.array(f['images'])
        self.lab_set = np.array(f['labels'])

        self.num = self.im_set.shape[0]
        assert self.im_set.shape[0]==self.lab_set.shape[0], '#images and #labels do not match!'
        assert self.im_set.shape[1]==self.load_size, 'Image size error!'
        assert self.im_set.shape[2]==self.load_size, 'Image size error!'
        print('# Images found:', self.num)

        self.shuffle()
        self._idx = 0

    def next_batch(self, batch_size):
        labels_batch = np.zeros(batch_size)
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3))

        for i in range(batch_size):
            image = self.im_set[self._idx]
            image = image.astype(np.float32)/255. - self.data_mean
            if self.randomize:
                trans = np.random.random_integers(-1, 1)
                # if trans>0:
                    # image = image[:,::-1,:]
                image = cv2.trans(image, trans)
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            images_batch[i, ...] = image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            labels_batch[i, ...] = self.lab_set[self._idx]

            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
                if self.randomize:
                    self.shuffle()

        return images_batch, labels_batch

    def size(self):
        return self.num

    def reset(self):
        self._idx = 0

    def shuffle(self):
        perm = np.random.permutation(self.num)
        self.im_set = self.im_set[perm]
        self.lab_set = self.lab_set[perm]

# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.perm = kwargs['perm']
        self.data_root = os.path.join(kwargs['data_root'])

        # read data info from lists
        self.list_im = []
        self.list_lab = []
        with open(kwargs['data_list'], 'r') as f:
            for line in f:
                l =line.rstrip().split(' ')
                path, lab = l[0], l[1]
                self.list_im.append(os.path.join(self.data_root, path))
                self.list_lab.append(int(lab))
        self.list_im = np.array(self.list_im, np.object)
        self.list_lab = np.array(self.list_lab, np.int64)
        self.num = self.list_im.shape[0]
        print('# Images found:', self.num)

        # permutation
        if self.perm:
            perm = np.random.permutation(self.num)
            self.list_im[:, ...] = self.list_im[perm, ...]
            self.list_lab[:] = self.list_lab[perm, ...]

        self._idx = 0

    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3))
        labels_batch = np.zeros(batch_size)
        for i in range(batch_size):
            image = scipy.misc.imread(self.list_im[self._idx])
            image = scipy.misc.imresize(image, (self.load_size, self.load_size))
            image = image.astype(np.float32)/255.
            image = image - self.data_mean
            if self.randomize:
                trans = np.random.random_integers(0, 1)
                if trans>0:
                    image = image[:,::-1,:]
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)

                # trans = np.random.random_integers(0, 1)
                # cols,rows = 256,256
                # if trans == 1:
                    # image = cv2.flip(image, 0)
                # # elif trans == 0:
                    # # pass
                # # else:
                    # # p = np.random.random() * 2 -1
                    # # M = cv2.getRotationMatrix2D((cols/2,rows/2),20*p,1)
                    # # image = cv2.warpAffine(image,M,(cols,rows))
                # offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                # offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            images_batch[i, ...] =  image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            labels_batch[i, ...] = self.list_lab[self._idx]

            self._idx += 1
            if self._idx == self.num:
                self._idx = 0

        return images_batch, labels_batch

    def size(self):
        return self.num

    def reset(self):
        self._idx = 0
