import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors
from skimage import io
from math import sin, cos, pi




# heatmap augmentation parameters
flip = 0.5
rotate = 10
scale = 1
translate = 0
left_right_swap = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# size of text encoding
sentence_vector_size = 768 #300

# weight of text encoding
encoding_weight = 30

# size of compressed text encoding
compress_size = 300 #128

# text encoding interpolation
beta = 0.5

# numbers of channels of the convolutions
convolution_channel_g = [256, 128, 64, 32]
convolution_channel_d = [32, 64, 128, 256]

# hidden layers
hidden = [250, 200]

# coordinate scaling
factor = 20

# visible threshold
v_threshold = 0.8

noise_size = 128
g_input_size = noise_size + compress_size
d_final_size = convolution_channel_d[-1]

total_keypoints = 17
# to decide whether a keypoint is in the heatmap
heatmap_threshold = 0.2

# have more than this number of keypoints to be included
keypoint_threshold = 7

# get a batch of noise vectors




# generator given noise input
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # several layers of transposed convolution, batch normalization and ReLu
        self.first = nn.ConvTranspose2d(noise_size, convolution_channel_g[0], 4, 1, 0, bias=False)
        self.main = nn.Sequential(
            nn.BatchNorm2d(convolution_channel_g[0]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel_g[0], convolution_channel_g[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_g[1]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel_g[1], convolution_channel_g[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_g[2]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel_g[2], convolution_channel_g[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_g[3]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel_g[3], total_keypoints, 4, 2, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, noise_vector):
        return self.main(self.first(noise_vector))


# discriminator given heatmap
class Discriminator(nn.Module):
    def __init__(self, bn=False, sigmoid=False):
        super(Discriminator, self).__init__()

        # several layers of convolution and leaky ReLu
        self.main = nn.Sequential(
            nn.Conv2d(total_keypoints, convolution_channel_d[0], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_d[0]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel_d[0], convolution_channel_d[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_d[1]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel_d[1], convolution_channel_d[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_d[2]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel_d[2], convolution_channel_d[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_d[3]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)

        )
        self.second = nn.Conv2d(convolution_channel_d[-1], d_final_size, 1, bias=False)
        self.third = nn.Sequential(
            nn.BatchNorm2d(d_final_size) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_final_size, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() if sigmoid else nn.Identity()

        )

    def forward(self, input_heatmap):
        return self.third(self.second(self.main(input_heatmap)))


# generator given noise and sentence input (regression)
class Generator_R(nn.Module):
    def __init__(self):
        super(Generator_R, self).__init__()

        # several layers of FC and ReLu
        self.main = nn.Sequential(
            nn.Linear(noise_size + sentence_vector_size, hidden[0], bias=True),
            nn.ReLU(True),
            nn.Linear(hidden[0], hidden[1], bias=True),
            nn.ReLU(True),
            nn.Linear(hidden[1], total_keypoints * 3, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, noise_vector, sentence_vector):
        output = self.main(torch.cat((noise_vector, sentence_vector), 1))
        return torch.cat(
            (output[:, 0:total_keypoints * 2] * factor,
             self.sigmoid(output[:, total_keypoints * 2:total_keypoints * 3])),
            1)


# discriminator given coordinates and sentence input (regression)
class Discriminator_R(nn.Module):
    def __init__(self):
        super(Discriminator_R, self).__init__()

        # several layers of FC and ReLu
        self.main = nn.Sequential(
            nn.Linear(total_keypoints * 3 + sentence_vector_size, hidden[0], bias=True),
            nn.ReLU(True),
            nn.Linear(hidden[0], hidden[1], bias=True),
            nn.ReLU(True),
            nn.Linear(hidden[1], 1, bias=True)
        )

    def forward(self, input_coordinates, sentence_vector):
        input = torch.cat((input_coordinates[:, 0:total_keypoints * 2] / factor,
                           input_coordinates[:, total_keypoints * 2:total_keypoints * 3], sentence_vector), 1)
        return self.main(input)


# custom weights initialization called on net_g and net_d
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.normal_(m.bias.data, 0.0, 0.02)


# generator given noise and text encoding input
class Generator2(Generator):
    def __init__(self):
        super(Generator2, self).__init__()

        self.first2 = nn.ConvTranspose2d(g_input_size, convolution_channel_g[0], 4, 1, 0, bias=False)

        # compress text encoding first
        self.compress = nn.Sequential(
            nn.Linear(sentence_vector_size, compress_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, noise_vector, sentence_vector):
        # concatenate noise vector and compressed sentence vector
        input_vector = torch.cat((noise_vector, (
            (self.compress(sentence_vector.view(-1, sentence_vector_size))).view(-1, compress_size, 1, 1))), 1)

        return self.main(self.first2(input_vector))


# discriminator given heatmap and sentence vector
class Discriminator2(Discriminator):
    def __init__(self, bn=False, sigmoid=False):
        super(Discriminator2, self).__init__(bn, sigmoid)

        # convolution with concatenated sentence vector
        self.second2 = nn.Conv2d(convolution_channel_d[-1] + compress_size, d_final_size, 1, bias=False)

        # compress text encoding first
        self.compress = nn.Sequential(
            nn.Linear(sentence_vector_size, compress_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input_heatmap, sentence_vector):
        # first convolution, then concatenate sentence vector
        tensor = torch.cat((self.main(input_heatmap), (
            (self.compress(sentence_vector.view(-1, sentence_vector_size))).view(-1, compress_size, 1, 1)).repeat(1, 1,
                                                                                                                  4,
                                                                                                                  4)),
                           1)
        return self.third(self.second2(tensor))

