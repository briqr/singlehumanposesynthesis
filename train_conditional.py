#from model_modular import *
from model.model import *
import torch.optim as optim
from datetime import datetime
#from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad
from datasets.coco_dataset import *
import datasets.custom_transforms as custom_transforms
import torchvision.transforms as transforms
from datasets.feature_processing import HeatmapGenerator
from model.model_util import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from easydict import EasyDict as edict
import yaml
import vis

class GanTrainer():

    def __init__(self, params):

        self.params = params
        num_keypoints = params.data.num_keypoints

        workers = 8
        # training parameters
        batch_size = 128
        learning_rate_g = 0.0004
        learning_rate_d = 0.0004

        # ADAM solver
        beta_1 = 0.0
        beta_2 = 0.9
        # load text encoding model

        res = params.train.resolution
        with open(params_path, 'r') as stream:
            params = edict(yaml.load(stream))
        image_pose_transform = transforms.Compose([
            custom_transforms.Resize((res, res)),
            # custom_transforms.RandomCrop(params.train.resolution),
            custom_transforms.ToTensor(),
            custom_transforms.Normalize((0, 0, 0), (1, 1, 1))
        ])
        heatmap_generator = HeatmapGenerator(resolution=(res, res), num_keypoints=num_keypoints)
        dataset = CocoDataset(params, 'train', image_pose_transform, heatmap_generator) #, transform=None, feature_generator)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_sample)



        self.net_g = Generator2().to(device)
        self.net_d = Discriminator2().to(device)
        self.net_g.apply(weights_init)
        self.net_d.apply(weights_init)

        self.optimizer_g = optim.Adam(self.net_g.parameters(), lr=learning_rate_g, betas=(beta_1, beta_2))
        self.optimizer_d = optim.Adam(self.net_d.parameters(), lr=learning_rate_d, betas=(beta_1, beta_2))

    def train(self):
        # penalty coefficient
        lamb = 150

        # level of text-image matching
        alpha = 1

        # train discriminator k times before training generator
        k = 5
        self.net_g.train()
        self.net_d.train()
        # train
        start = datetime.now()
        print(start)
        print('training')
        self.net_g.train()
        self.net_d.train()
        iteration = 1
        #writer = SummaryWriter(comment='_caption')
        loss_g = torch.tensor(0)
        beta = 0.5
        noise_size = self.params.params.noise_size
        # number of batches
        batch_number = len(self.data_loader)
        num_epochs = 1200
        fixed_noise = get_noise_tensor(10, noise_size).to(device)
        fixed_text_enc = None
        for epoch in range(0, num_epochs):
            print('learning rate: g ' + str(self.optimizer_g.param_groups[0].get('lr')) + ' d ' + str(
                self.optimizer_d.param_groups[0].get('lr')))

            for i, batch in enumerate(self.data_loader, 0):
                # first, optimize discriminator
                self.net_d.zero_grad()

                heatmap_real = batch.get('heatmap')
                text_encoding = batch['text_encoding']
                text_interpolated1 = batch['interpolation_text_encoding1']
                text_interpolated2 = batch['interpolation_text_encoding2']
                text_interpolated = beta * text_interpolated1 + (1 - beta) * text_interpolated2
                text_match = text_encoding.to(device).unsqueeze(-1).unsqueeze(-1)
                text_interpolated = text_interpolated.to(device).unsqueeze(-1).unsqueeze(-1)
                text_mismatch = text_interpolated1.to(device).unsqueeze(-1).unsqueeze(-1)

                current_batch_size = len(heatmap_real)

                noise = get_noise_tensor(current_batch_size, noise_size)

                heatmap_real = heatmap_real.to(device)


                noise = noise.to(device)

                # discriminate heatmpap-text pairs
                score_right = self.net_d(heatmap_real, text_match)
                score_wrong = self.net_d(heatmap_real, text_mismatch)

                # generate heatmaps
                heatmap_fake = self.net_g(noise, text_match).detach()

                # discriminate heatmpap-text pairs
                score_fake = self.net_d(heatmap_fake, text_match)

                # random sample
                epsilon = np.random.rand(current_batch_size)
                heatmap_sample = torch.empty_like(heatmap_real)
                for j in range(current_batch_size):
                    heatmap_sample[j] = epsilon[j] * heatmap_real[j] + (1 - epsilon[j]) * heatmap_fake[j]
                heatmap_sample.requires_grad = True
                text_match.requires_grad = True

                # calculate gradient penalty
                score_sample = self.net_d(heatmap_sample, text_match)
                gradient_h, gradient_t = grad(score_sample, [heatmap_sample, text_match], torch.ones_like(score_sample),
                                              create_graph=True)
                gradient_norm = (gradient_h.pow(2).sum((1, 2, 3)) + gradient_t.pow(2).sum((1, 2, 3))).sqrt()
                #print('888gradient norm shape', gradient_norm.shape)

                # calculate losses and update
                loss_d = (score_fake + alpha * score_wrong - (1 + alpha) * score_right + lamb * (
                    torch.max(torch.tensor(0, dtype=torch.float32, device=device), gradient_norm - 1).pow(2))).mean()
                loss_d.backward()
                self.optimizer_d.step()

                # log
                #writer.add_scalar('loss/d', loss_d, batch_number * (epoch) + i)

                # second, optimize generator
                if iteration == k:
                    self.net_g.zero_grad()
                    iteration = 0

                    # get sentence vectors and noises
                    #text_interpolated = dataset.get_interpolated_caption_tensor(current_batch_size)
                    noise = get_noise_tensor(current_batch_size, noise_size)
                    noise2 = get_noise_tensor(current_batch_size, noise_size )
                    noise = noise.to(device)
                    noise2 = noise2.to(device)

                    # generate heatmaps
                    heatmap_fake = self.net_g(noise, text_match)
                    heatmap_interpolated = self.net_g(noise2, text_interpolated)

                    # discriminate heatmpap-text pairs
                    score_fake = self.net_d(heatmap_fake, text_match)
                    score_interpolated = self.net_d(heatmap_interpolated, text_interpolated)

                    # discriminate losses and update
                    loss_g = -(score_fake + score_interpolated).mean()
                    loss_g.backward()
                    self.optimizer_g.step()

                    # log
                    #writer.add_scalar('loss/g', loss_g, batch_number * (epoch) + i)

                # print progress
                print('epoch ' + str(epoch + 1) + ' batch ' + str(i + 1) + ' of ' + str(
                    batch_number) + ' g loss: ' + str(loss_g.item()) + ' d loss: ' + str(loss_d.item()))

                iteration = iteration + 1
                #break
            freq = 10
            if epoch % freq == 0:
                self.net_d.eval()
                self.net_g.eval()
                if fixed_text_enc is None:
                    fixed_text_enc = text_match[:len(fixed_noise)]
                    vis.save_heatmap(heatmap_real[:len(fixed_noise)], epoch, title='real',
                                     res_path='training_results_pose')
                heatmap_fake = self.net_g(fixed_noise, fixed_text_enc).detach()
                vis.save_heatmap(heatmap_fake, epoch, res_path='training_results_pose')

                self.net_d.train()
                self.net_g.train()

            # save models
            model_path = params.model.gan.model_path
            torch.save(self.net_g.state_dict(), os.path.join(model_path, 'pose_generator_epoch%d') %epoch)
            torch.save(self.net_d.state_dict(), os.path.join(model_path, 'pose_discriminator_epoch%d' %epoch))


        print('\nfinished')
        print(datetime.now())
        print('(started ' + str(start) + ')')
        #writer.close()



if __name__ == '__main__':
    params_path = 'config/coco_train_config.yml'
    with open(params_path, 'r') as stream:
        params = edict(yaml.load(stream))

    gan_trainer = GanTrainer(params)
    gan_trainer.train()

