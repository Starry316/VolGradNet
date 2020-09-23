"""
This code is base on the source code of
GradNet: Unsupervised Deep Screened Poisson Reconstruction for Gradient-Domain Rendering
Credit to Guo et al.
"""
import torch
from networks import GradGenNet


class GradGenerator():
    def __init__(self, args, loss_func, feature_channel=12):
        self.net = GradGenNet(feature_channel = feature_channel)
        self.net = torch.nn.DataParallel(self.net)
        self.loss_func = loss_func
        if args.cuda:
            self.net.cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, betas=(args.beta, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, 0.95)

    def set_training(self, training=True):
        if training:
            self.net.train()
        else:
            self.net.eval()

    def scheduler_step(self):
        self.scheduler.step()

    def predict(self, net_input):
        return self.net(net_input)

    def unfreeze(self):
        for param in self.net.parameters():
            param.requires_grad = True

    def freeze(self):
        for param in self.net.parameters():
            param.requires_grad = False

    def update(self, nI, grad, feature, I):
        # clear grad
        self.optimizer.zero_grad()
        loss = self.loss_func.eval_feature_loss(nI.detach(), grad, feature, I)
        loss.backward()
        # update params
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.net.state_dict(), path)
