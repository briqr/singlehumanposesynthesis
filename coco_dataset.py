# for bert model
import cv2
import torch
import torch.utils.data
from pycocotools.coco import COCO
import os
#import imageio as io
import numpy as np
import pickle
from math import sin, cos, pi
from PIL import Image
from collections import defaultdict
import random
from transformers import *
MAX_NUM_WORDS = 12

heatmap_size = 64

x_grid = np.repeat(np.array([range(heatmap_size)]), heatmap_size, axis=0)
y_grid = np.repeat(np.array([range(heatmap_size)]).transpose(), heatmap_size, axis=1)
empty = np.zeros([heatmap_size, heatmap_size], dtype='float32')
left_right_swap = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

class CocoDataset(torch.utils.data.Dataset):
    """Coco Landmarks dataset."""

    def __init__(self, params, subset_type, transform = None, feature_gen=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.path = params.data.root_path
        self.img_dir = '%s/images/%s2017' %(self.path, subset_type)
        self.annotation_path = 'data/coco_%s.pkl' % subset_type
        no_encoding = True
        to_open_path = self.annotation_path #annotation_path_with_encoding
        if False: #not os.path.exists(self.annotation_path):
        #if not os.path.exists(annotation_path_with_encoding):
            #os.mkdir('data/')
            self.prepare_data(subset_type)
            no_encoding = True
            to_open_path = self.annotation_path
        annotation_path_with_encoding = 'data/coco_densepose_%s_withencoding.pkl'%subset_type
        with open (annotation_path_with_encoding, 'rb') as f:
            self.data = pickle.load(f)
            self.img_ids = list(self.data.keys())
        if False:
            bs = 200
            for b in range(len(self.img_ids) // bs + 176):
                self.prepare_text_encoding(annotation_path_with_encoding, b, bs)
            #self.prepare_text_encoding(annotation_path_with_encoding)
            with open(annotation_path_with_encoding, 'rb') as f:
                self.data = pickle.load(f)
                self.img_ids = list(self.data.keys())
            #print(self.img_ids)
        print('init len of img ids', len(self.img_ids))
        new_im_ids = []
        for img_id in self.img_ids:
            keypoints = np.asarray(self.data[img_id]['keypoints'])
            if(len(keypoints) ==1 and np.sum(keypoints[:,:,2]==0)<11):
                new_im_ids.append(img_id)

        self.img_ids = new_im_ids
        self.transform = transform
        self.heatmap_generator = feature_gen

    def __len__(self):
        print('number of image samples with pesons in them', len(self.img_ids))
        #return 204
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        filename = self.data[img_id]['file_name']
        text = self.data[img_id]['text']
        text_encoding = random.choice(self.data[img_id]['text_encoding'])
        rnd_index = random.randint(0, len(self.img_ids)-1)
        interpolated_text1\
            = random.choice(self.data[self.img_ids[rnd_index]]['text_encoding'])
        rnd_index = random.randint(0, len(self.img_ids) - 1)
        interpolated_text2 = random.choice(self.data[self.img_ids[rnd_index]]['text_encoding'])
        keypoints = np.asarray(self.data[img_id]['keypoints'])

        #x = keypoints[:,:,1].copy()
        #keypoints[:,:,1] = keypoints[:, :, 0]
        #keypoints[:, :, 0] = x
        img_path = os.path.join(self.img_dir, filename)
        #image = io.imread(img_path)
        #image = Image.open(img_path).convert('RGB')

        #sample = {'image': image, 'text': text}
        if False:
            sample = {'keypoints': keypoints, 'text': text, 'interpolation_text1': interpolated_text1,
                      'interpolation_text2': interpolated_text2}

            heatmap_size = 64
            x0, y0, w, h = tuple(self.data[img_id]['bbox'][0])
            heatmap = np.empty((17, heatmap_size, heatmap_size), dtype='float32')

            # keypoints location (x, y) and visibility (v)
            x, y, v = get_coordinates(x0, y0, w, h, keypoints)

            # do heatmap augmentation
            augment = True
            total_keypoints = 17
            heatmap_size = 64
            flip = 0.5
            rotate = 10
            sigma = 2
            scale = 1
            translate = 0

            if augment:
                # random flip, rotation, scaling, translation
                f, a, s, tx, ty = get_augment_parameters(flip, scale, rotate, translate)
                x, y, v = augment_heatmap(x, y, v, heatmap_size / 2, f, a, s, tx, ty)

            for i in range(total_keypoints):
                # labeled keypoints' v > 0
                if v[i] > 0:
                    # ground truth in heatmap is normal distribution shaped
                    heatmap[i] = np.exp(-((x_grid - x[i]) ** 2 + (y_grid - y[i]) ** 2) / (2 * sigma ** 2),
                                        dtype='float32')
                else:
                    heatmap[i] = empty.copy()

            sample['heatmap'] = torch.tensor(heatmap * 2 - 1, dtype=torch.float32)#.unsqueeze(0)

        else:
            #img_path = os.path.join(self.img_dir, filename)
            #image = Image.open(img_path).convert('RGB')
            bbox = self.data[img_id]['bbox'][0]
            #image = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            keypoints[:, :, 0] = keypoints[:, :, 0] - bbox[0]
            keypoints[:, :, 1] = keypoints[:, :, 1] - bbox[1]
            #sample = {'image': image, 'keypoints': keypoints, 'text': text}
            sample = {'keypoints': keypoints, 'text': text, 'text_encoding': text_encoding, 'interpolation_text_encoding1': interpolated_text1,
                      'interpolation_text_encoding2': interpolated_text2}
            sample['orig_shape'] = [bbox[3], bbox[2]]
            if self.transform:
                sample = self.transform(sample)
                #sample['image'] = self.transform(sample['image'])
                #del sample['text']
            del sample['orig_shape']
            if self.heatmap_generator is not None:
                sample['heatmap'] = self.heatmap_generator(sample['keypoints']) * 2 - 1
                #        import matplotlib.pyplot as plt; hm = sample['heatmap'] ;plt.imshow(hm.sum(dim=0).cpu().numpy()); plt.show()
            if False and sample['heatmap'].sum()>0:
                #visual_im(sample['image'].permute(1, 2, 0), 'im' + str(img_id))
                heatmap = torch.sum(sample['heatmap'], dim=0)
                visual_im(heatmap, 'hm'+str(img_id))
                import matplotlib.pyplot as plt
                plt.imshow(sample['image'].permute(1,2, 0).cpu().numpy())
                plt.show()
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            #import matplotlib.pyplot as plt; hm = sample['heatmap'] ;plt.imshow(hm.sum(dim=0).cpu().numpy()); plt.show()

        return sample


    def prepare_text_encoding(self, ann_path, bi, bs):
        pretrained_weights = 'bert-base-uncased'
        text_tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        text_model = BertModel.from_pretrained(pretrained_weights)
        i = 0
        currentimg_ids = self.img_ids[bi * bs:(bi + 1) * bs]
        if bi > 0:
            with open(ann_path, 'rb') as f:
                self.data = pickle.load(f)
        for img_id in currentimg_ids:
            sample = self.data[img_id]
            with torch.no_grad():
                print('encoding for img id: ', i)
                text_encodings = encode_text(text_tokenizer, text_model, sample['text'])
            sample['text_encoding'] = text_encodings
            i += 1
            #if i ==4 :
                #break
        with open(ann_path, 'wb') as data_file:
            pickle.dump(self.data, data_file)


    def prepare_data(self, subset_type):

        self.annot_dir = '%s/annotations_trainval2017/annotations' % self.path
        self.annotation_file = '%s/person_keypoints_%s2017.json' % (self.annot_dir, subset_type) #for person keypoint subset 2017
        annotation_file = self.annotation_file

        if False and subset_type != 'train':
            densepose_subsettype = 'valminusminival'
            annotation_file = '/media/datasets_local/DensePose_COCO/densepose_coco_2014_%s.json' % densepose_subsettype

        #self.annotation_file = '/home/briq/datasets/coco2014/annotations_trainval2014/annotations/person_keypoints_%s2014.json' % (subset_type)
#        self.annotation_file = '%s/instances_%s2017.json' % (self.annot_dir, subset_type) #for instance subset
        self.caption_file = '%s/captions_%s2017.json' % (self.annot_dir, subset_type) #2017 dataset

        self.coco_set = COCO(annotation_file)
        self.coco_text = COCO(self.caption_file)

        catIds = self.coco_set.getCatIds(catNms=['person'])
        self.img_ids = self.coco_set.getImgIds(catIds=catIds) # for person subset
        #self.img_ids = self.coco_set.getImgIds() # for instance subset
        imgId_annot = dict()
        image_info = self.coco_set.dataset['images']
        for im_info in image_info:
            img_id = im_info['id']
            if not img_id in self.img_ids:
                continue
            anns = self.coco_set.imgToAnns[img_id]
            current_text = self.coco_text.imgToAnns[img_id] # for coco 2017

            #current_text = None # for coco 2014
            #imgId_annot[img_id] = []
            #for ann in anns:
            current_keypoints = []
            current_segmentations = []
            current_bboxes = []
            densepose_xy = []
            densepose_uv = []
            im_height, im_width = im_info['height'], im_info['width']


            for ann in anns:

                if ('dp_masks' in ann.keys()):

                    bounding_box = np.round(bbox)
                    Point_x = np.array(ann['dp_x']) / 255. * bounding_box[2]  # Strech the points to current box.
                    Point_y = np.array(ann['dp_y']) / 255. * bounding_box[3]  # Strech the points to current box.
                    Point_I = np.array(ann['dp_I'])
                    Point_U = np.array(ann['dp_U'])
                    Point_V = np.array(ann['dp_V'])
                    x1, y1, x2, y2 = bounding_box[0], bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + \
                                     bounding_box[3]
                    x2 = min([x2, im_height])
                    y2 = min([y2, im_width])
                    Point_x = Point_x + x1
                    Point_y = Point_y + y1
                    densepose_uv.append([Point_I, Point_U, Point_V])
                    densepose_xy.append([Point_x, Point_y])


                keypoints = np.asarray(ann['keypoints']).reshape(17,3)
                bbox = ann['bbox']
                segmentation = ann['segmentation']
                current_keypoints.append(keypoints)
                current_segmentations.append(segmentation)
                current_bboxes.append(bbox)

            #image_ann_object = {'file_name': im_info['file_name'], 'keypoints': current_keypoints, 'bbox': current_bboxes, 'segmentation': current_segmentations, 'text': current_text}
            #image_ann_object = {'file_name': im_info['file_name'], 'bbox': current_bboxes, 'segmentation': current_segmentations, 'text': current_text}
            image_ann_object = {'densepose_uv': densepose_uv, 'densepose_xy': densepose_xy,
                                'file_name': im_info['file_name'], 'height': im_height, 'width': im_width,
                                'keypoints': current_keypoints, 'bbox': current_bboxes,
                                'segmentation': current_segmentations, 'text': current_text}

            imgId_annot[img_id] = image_ann_object #.append(ann)

        with open(self.annotation_path, 'wb') as data_file:
            pickle.dump(imgId_annot, data_file)

        def get_caption(self, img_id):
            # a list of indices for a sentence
            sent_caption = np.asarray(self.captions[img_id]).astype('int64')
            if (sent_caption == 0).sum() > 0:
                print('ERROR: do not need END (0) token', sent_caption)
            num_words = len(sent_caption)
            # pad with 0s (i.e., '<end>')
            x = np.zeros((MAX_NUM_WORDS, 1), dtype='int64')
            x_len = num_words
            if num_words <= MAX_NUM_WORDS:
                x[:num_words, 0] = sent_caption
            else:
                ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
                np.random.shuffle(ix)
                ix = ix[:MAX_NUM_WORDS]
                ix = np.sort(ix)
                x[:, 0] = sent_caption[ix]
                x_len = MAX_NUM_WORDS
            return x, x_len

        def build_dictionary(self, train_captions, test_captions):
            word_counts = defaultdict(float)
            captions = train_captions + test_captions
            for sent in captions:
                for word in sent:
                    word_counts[word] += 1

            vocab = [w for w in word_counts if word_counts[w] >= 0]

            ixtoword = {}
            ixtoword[0] = '<end>'
            wordtoix = {}
            wordtoix['<end>'] = 0
            ix = 1
            for w in vocab:
                wordtoix[w] = ix
                ixtoword[ix] = w
                ix += 1

            train_captions_new = []
            for t in train_captions:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                # rev.append(0)  # do not need '<end>' token
                train_captions_new.append(rev)

            test_captions_new = []
            for t in test_captions:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                # rev.append(0)  # do not need '<end>' token
                test_captions_new.append(rev)

            return [train_captions_new, test_captions_new,
                    ixtoword, wordtoix, len(ixtoword)]

    # from stack gan, in case i decide to use their text encoder: https://github.com/hanzhanggit/StackGAN-Pytorch
    def load_wordembeddings(self):
        embedding_filename = 'data/coco/train/char-CNN-RNN-embeddings.pickle'
        with open(embedding_filename, 'rb') as f:
            import codecs
            pickled = codecs.encode(f, "utf-8").decode()
            print('pickled', pickled)
            self.embeddings = pickle.load(f)
            self.embeddings = np.array(self.embeddings)

        filepath = os.path.join('data/coco/train/filenames.pickle')
        with open(filepath, 'rb') as f:
            self.filenames = pickle.load(f)
            print('filenames', self.filenames)


def encode_text(text_tokenizer, text_model, text):

    encodings = []
    #separate_word_encoding = []
    raw_text = []
    for i in range(len(text)):
        if True: # if bert model
            current_text = text[i]['caption']
            tokens = torch.tensor([text_tokenizer.encode(current_text, add_special_tokens=True)])
            text_encoding = text_model(tokens)[0]
            #text_encoding = torch.mean(text_encoding, dim=1) #average the encodings across the words
            encodings.append(text_encoding)
        # print('***len token, text encoding shape', len(tokens), text_encoding.shape)
        #fasttext
        else:
            text_encoding = text_model.get_sentence_vector(current_text.replace('\n', '').lower()) * encoding_weight
            encodings.append(torch.from_numpy(text_encoding).cuda().unsqueeze(-1).unsqueeze(-1))
        # print('current text: ', current_text)
        # print('func text encodig size: ', text_encoding.shape)

        raw_text.append(current_text)
    #encodings = torch.stack(encodings)
    return encodings

def visual_im(im, title):
    #im_ar = np.array(im)
    im_ar = im.cpu().numpy()
    cv2.imshow(title, im_ar)
    #cv2.waitKey(0)


def collate_sample(samples):
    keys = samples[0].keys()
    all_samples = dict()
    for key in keys:
        all_samples[key] = []

    for sample in samples:
        for key in keys:
            all_samples[key].append(sample[key])

    keys = ['image', 'heatmap', 'text_encoding', 'interpolation_text_encoding1', 'interpolation_text_encoding2']
    for key in keys:
        if key in all_samples:
            if 'text' in key:
                all_samples[key] = torch.stack(all_samples[key])
                all_samples[key] = all_samples[key][:,0]

            else:
                all_samples[key] = torch.stack(all_samples[key])
    # total_hm = torch.sum(all_samples[key][0], dim=0);
    # import cv2;
    # cv2.imshow('hm', total_hm.cpu().numpy() * 255);
    # cv2.waitKey(0)
    # for key in keys:
    #     if key != 'text':
    #         print('key', key)
    #         if len(all_samples[key][0].shape) == 3:
    #             all_samples[key] = torch.cat(all_samples[key], 0).view(len(all_samples[key]), all_samples[key][0].shape[0], all_samples[key][0].shape[1], all_samples[key][0].shape[2])
    #         else:
    #             all_samples[key] = torch.cat(all_samples[key], 0).view(len(all_samples[key]), all_samples[key][0].shape[0], all_samples[key][0].shape[1])
    return all_samples



def get_coordinates(x0, y0, w, h, keypoint):
    heatmap_size = 64
    # keypoints location (x, y) and visibility (v, 0 invisible, 1 visible)
    x = keypoint[:, :, 0][0]
    y = keypoint[:, :, 1][0]
    v = keypoint[:, :, 2][0].clip(0, 1)

    # calculate the scaling
    heatmap_half = heatmap_size / 2
    if h > w:
        x = heatmap_half - w / h * heatmap_half + (x - x0) / h * heatmap_size
        y = (y - y0) / h * heatmap_size
    else:
        x = (x - x0) / w * heatmap_size
        y = heatmap_half - h / w * heatmap_half + (y - y0) / w * heatmap_size

    # set invisible keypoint coordinates as (0,0)
    x[v < 1] = 0
    y[v < 1] = 0

    return x, y, v

def get_augment_parameters(flip, scale, rotate, translate):
    # random flip, rotation, scaling, translation
    import random
    from math import pi
    f = random.random() < flip
    a = random.uniform(-rotate, rotate) * pi / 180
    s = random.uniform(scale, 1 / scale)
    tx = random.uniform(-translate, translate)
    ty = random.uniform(-translate, translate)
    return f, a, s, tx, ty

    # do heatmap augmentation
def augment_heatmap(x, y, v, heatmap_half, f, a, s, tx, ty):
    x = x - heatmap_half
    y = y - heatmap_half

    # flip
    if f:
        x = -x

        # when flipped, left and right should be swapped
        x = x[left_right_swap]
        y = y[left_right_swap]
        v = v[left_right_swap]

    # rotation
    sin_a = sin(a)
    cos_a = cos(a)
    x, y = tuple(np.dot(np.array([[cos_a, -sin_a], [sin_a, cos_a]]), np.array([x, y])))

    # scaling
    x = x * s
    y = y * s

    # translation
    x = x + tx + heatmap_half
    y = y + ty + heatmap_half

    return x, y, v

if __name__ == '__main__':
    coco_ds = CocoDataset('/media/datasets/pose_estimation/MSCOCO_2017', 'train')
    item = coco_ds.__getitem__(1)
    print(item)


