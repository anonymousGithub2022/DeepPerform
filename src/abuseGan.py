import torch
import torch.nn as nn
from tqdm import tqdm
import os
import torch.nn.functional as F
import copy


class SelfAttn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(SelfAttn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # 8*13*13
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16*5*5
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
            # 32*1*1
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output


class Generator(nn.Module):
    def __init__(self, gen_input_nc, image_nc):
        super(Generator, self).__init__()

        encoder_lis = [
            nn.Conv2d(gen_input_nc, 64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            # 8*26*26
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(),
            # 16*12*12
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(),
            # 32*5*5
        ]
        layer1 = [ResnetBlock(256, 512)]
        layer2 = [ResnetBlock(512, 512)]
        layer3 = [ResnetBlock(512, 512)]
        layer4 = [ResnetBlock(512, 256)]

        bottle_neck_lis = layer1 + layer2 + layer3 + layer4

        decoder_lis = [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(),

            SelfAttn(128, 'relu'),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(),

            SelfAttn(64, 'relu'),

            nn.ConvTranspose2d(64, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. image_nc x 28 x 28
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)


    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(in_dim, out_dim, padding_type, norm_layer, use_dropout, use_bias)
        if in_dim != out_dim:
            downsample = [
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=False),
                norm_layer(out_dim),
            ]
            self.downsample = nn.Sequential(*downsample)
        else:
            self.downsample = None

    def build_conv_block(self, in_dim, out_dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(out_dim),
            nn.LeakyReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.1)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(out_dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        if hasattr(self, 'downsample'):
            if self.downsample is not None:
                out = self.downsample(x) + self.conv_block(x)
                return out
            else:
                out = x + self.conv_block(x)
                return out
        else:
            out = x + self.conv_block(x)
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AbuseGanTransform:
    def __init__(self, model, image_nc, pert_lambda, per_size, adv_opt, attack_norm, device):

        super(AbuseGanTransform, self).__init__()
        self.device = device
        self.adv_opt = adv_opt
        self.per_size = per_size
        self.model = model
        self.norm = attack_norm
        self.box_min = 0.0
        self.box_max = 1.0
        self.now_epoch = 0
        self.pert_lambda = pert_lambda

        self.state_dict = copy.deepcopy(model.state_dict())

        self.gen_input_nc = image_nc
        self.netG = Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)

    def train_batch(self, x, labels):
        self.model.load_state_dict(self.state_dict)
        perturbation = self.netG(x)
        # add a clipping trick
        if self.norm == float('inf'):
            perturbation = torch.clamp(perturbation, -self.per_size, self.per_size)
        else:
            ori_shape = perturbation.shape
            per_norm = perturbation.reshape([len(x), -1]).norm(p=self.norm, dim=-1)
            perturbation = (self.per_size / per_norm).reshape([len(x), -1]) * perturbation.reshape([len(x), -1])
            perturbation = perturbation.reshape(ori_shape)
        adv_images = perturbation + x
        adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

        # optimize D
        for i in range(1):
            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            # loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            # loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            # self.optimizer_D.step()

        gate_criterion = torch.nn.BCELoss()
        loss_gate = None
        loss_perturb = None
        # optimize G
        for i in range(1):
            # cal G's loss in GAN
            self.optimizer_G.zero_grad()

            # pred_fake = self.netDisc(adv_images)
            # loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            # loss_G_fake.backward(retain_graph=True)

            perturb = torch.norm(perturbation.view(perturbation.shape[0], -1), self.norm, dim=1)
            loss_perturb = torch.mean(perturb)

            # cal gate loss
            preds_, masks_, gate_probs = self.model(adv_images, self.device)

            if self.adv_opt:
                opt_gates = torch.ones_like(gate_probs, device=self.device)
            else:
                opt_gates = torch.zeros_like(gate_probs, device=self.device)

            loss_gate = [gate_criterion(gate_probs[i], opt_gates[i]) for i in range(len(adv_images))]
            loss_gate = sum(loss_gate) / len(adv_images)
            loss_G = loss_gate       #   + self.pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()
        return loss_gate.item(), loss_perturb.item()

    def train(self, data_dataloader, epochs, save_path):
        assert self.netG.training is True
        assert self.netDisc.training is True
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        loss_list = []
        for epoch in range(self.now_epoch, self.now_epoch + epochs):
            if self.now_epoch >= 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
            if self.now_epoch >= 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
            loss_perturb_sum = 0
            loss_gate_sum = 0
            for i, data in tqdm(enumerate(data_dataloader, start=0)):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_gate_batch, loss_perturb_batch = \
                    self.train_batch(images, labels)

                loss_perturb_sum += loss_perturb_batch
                loss_gate_sum += loss_gate_batch
            # print statistics
            num_batch = len(data_dataloader)
            loss_list.append((float(loss_perturb_sum / num_batch), float(loss_gate_sum / num_batch)))
            print("epoch %d: loss_perturb: %.3f, loss_gate: %.3f,\n" %
                  (epoch, loss_perturb_sum / num_batch, loss_gate_sum / num_batch))
            # save generator
            if self.now_epoch % 10 == 0:
                netG_file_name = os.path.join(save_path, 'netG_epoch_' + str(self.now_epoch) + '.pth')
                torch.save(self.netG.state_dict(), netG_file_name)
        self.now_epoch += epochs
        return loss_list

    def transform(self, images):
        self.netG.eval()
        perturbation = self.netG(images)

        if self.norm == float('inf'):
            images = torch.clamp(perturbation, -self.per_size, self.per_size) + images
        else:
            ori_shape = perturbation.shape
            per_norm = perturbation.reshape([len(images), -1]).norm(p=self.norm, dim=-1)
            per_norm = per_norm.reshape([len(images), -1])
            perturbation = perturbation.reshape([len(images), -1]) / per_norm * self.per_size
            perturbation = perturbation.reshape(ori_shape)
            images = perturbation + images
        self.netG.train()
        return torch.clamp(images, 0, 1)


# if __name__ == '__main__':
#     model = Generator(3, 3)
#     x = torch.zeros([1, 3, 32, 32])
#     y = model(x)
#     print()
