import numbers
import torchvision.transforms.functional as F
import random
from PIL import Image
import torch
from math import pi, sin, cos
import numpy as np
import cv2
class Resize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):

        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_pose):
        if 'image' in img_pose:
            img = img_pose['image'] # size is h,w
            orig_shape = img.size
            img = F.resize(img, self.size, self.interpolation)
            img_pose['image'] = img
        else:
            orig_shape = img_pose['orig_shape']

        if 'mask' in img_pose:
            mask = img_pose['mask']
            mask = cv2.resize(mask, dsize=(self.size[1], self.size[0]))
            img_pose['mask'] = mask


        pose = img_pose['keypoints']

        # import cv2;
        # import numpy as np;
        # cv2.imshow('img', np.asarray(img));
        # cv2.waitKey(0)

        pose[:,:, 0] = pose[:, :, 0] * self.size[0]/orig_shape[1]
        pose[:,:, 1] = pose[:, :, 1] * self.size[1]/orig_shape[0]


        img_pose['keypoints'] = pose

        if 'densepose_xy' in img_pose:
            img_pose['densepose_xy'][:, 0] = img_pose['densepose_xy'] [:,0] * self.size[0]/orig_shape[1]
            img_pose['densepose_xy'][:, 1] = img_pose['densepose_xy'][:, 1] * self.size[1] / orig_shape[0]
        return img_pose


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """


    def __call__(self, image_pose):
        if 'image' in image_pose:
            img = image_pose['image']
            img = F.to_tensor(img)
            image_pose['image'] = img

        if 'mask' in image_pose:
            mask = image_pose['mask']
            mask = torch.from_numpy(mask)
            image_pose['mask'] = mask

        if 'keypoints' in image_pose:
            image_pose['keypoints'] = torch.from_numpy(image_pose['keypoints'])
        if 'densepose_xy' in image_pose:
            image_pose['densepose_xy'] = torch.from_numpy(image_pose['densepose_xy'])
        return image_pose



class RandomCrop(object):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img_pose):
        #if True: #'image' in img_pose:
        img = img_pose['image']
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            orig_shape = img.size

        pose = img_pose['keypoints']


        i, j, h, w = self.get_params(img, self.size)
        #i, j, h, w = [89, 144, 256, 256]
        #print('crop params', i, j, h, w)
        img = F.crop(img, i, j, h, w)
        pose[:,:,0] = (pose[:,:,0] + i)#y coord
        pose[:,:,1] = (pose[:,:,1] + j)#x coord
        img_pose['image'] = img
        img_pose['keypoints'] = pose
        return img_pose

class RandomAugment():
    def __init__(self, resolution, flip, scale, rotate, translate):
        self.resolution = resolution
        self.flip = flip
        self.scale = scale
        self.rotate = rotate
        self.translate = translate
        self.left_right_swap  = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

    def __call__(self, image_pose):
        f = random.random() < self.flip
        a = random.uniform(-self.rotate, self.rotate) * pi / 180
        s = random.uniform(self.scale, 1 / self.scale)
        tx = random.uniform(-self.translate, self.translate)
        ty = random.uniform(-self.translate, self.translate)

        keypoints = image_pose['keypoints']
        x = keypoints[:,:,0] - self.resolution//2
        y = keypoints[:,:,1] - self.resolution//2

        # flip
        if f:
            x = -x
            # when flipped, left and right should be swapped
            x = x[self.left_right_swap]
            y = y[self.left_right_swap]
            keypoints[:, :, 2] = keypoints[:, :, 2][self.left_right_swap]

        # rotation
        sin_a = sin(a)
        cos_a = cos(a)
        x, y = tuple(np.dot(np.array([[cos_a, -sin_a], [sin_a, cos_a]]), np.array([x, y])))

        # scaling
        x = x * s
        y = y * s

        # translation
        x = x + tx + self.resolution/2
        y = y + ty + self.resolution/2
        keypoints[:, :, 0] = x
        keypoints[:, :, 1] = y

        image_pose['keypoints'] = keypoints
        #center = np.array((width / 2, height / 2))
        #mat = get_transform(center, s, self.resolution, a)[:2]
        #img = cv2.warpAffine(img, mat, self.resolution).astype(np.float32) / 255
        return image_pose


class Normalize(object):

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, image_pose):
        if 'image' in image_pose:
            img = image_pose['image']
            img = F.normalize(img, self.mean, self.std, self.inplace)
            image_pose['image'] = img
        return image_pose



def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def kpt_affine(kpt, mat):
    kpt = np.array(kpt)
    shape = kpt.shape
    kpt = kpt.reshape(-1, 2)
    return np.dot( np.concatenate((kpt, kpt[:, 0:1]*0+1), axis = 1), mat.T ).reshape(shape)
