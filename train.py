import argparse
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from dataset import Dataset
from networks import VolGradNet
from grad_gen import GradGenerator
from loss_functions import LossFunction
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

args = None
save_dir = 'vol/'
save_name = 'volGradNet-f12'
writer = SummaryWriter('logs/'+save_dir+ save_name)


def calc_loss(nI, I, dx, dy, feature, recon, grad):
    Grad_loss = loss_func.eval_grad_loss(nI, dx, dy)
    L1_loss = args.lambda_L1 * loss_func.eval_l1_loss(nI, I)
    Feature_loss = args.lambda_F * loss_func.eval_feature_loss(nI, grad.detach(), feature, recon)
    return L1_loss, Grad_loss, Feature_loss

def train_grad_generator(train_loader, epoch):
    g_loss = 0.0
    step = 0
    for I, dx, dy, feature, recon in train_loader:
        start_time = time.time()
        step += 1
        if args.cuda:
            I, dx, dy, feature, recon  = I.cuda(), dx.cuda(), dy.cuda(), feature.cuda(), recon.cuda()
        # image & feature
        I_input = torch.cat([I, feature], 1)
        # dx & dy
        grad_input = torch.cat([dx, dy], 1)
        grad = grad_generator.predict(torch.cat([I_input, grad_input], 1))
        loss = grad_generator.update(recon, grad, feature, recon)
        g_loss += loss
        print('step: %d/%d loss %f %fs' % (step,  len(train_loader), loss, time.time()-start_time))
    g_loss /= len(train_loader)
    writer.add_scalar('g loss', g_loss, epoch)
    print('Training --- g_loss: %f' % g_loss)

def train(network, optimizer, train_loader, epoch):
    network.train()
    grad_generator.set_training(False)
    train_loss = {'L1_loss': 0.0, 'Grad_loss': 0.0, 'Feature_loss': 0.0}
    step= 0
    # dataloader = DataPerfetcher(train_loader)
    # I, dx, dy, feature, recon = dataloader.next()
    # while I is not None:
    for I, dx, dy, feature, recon in train_loader:
        start_time = time.time()
        step += 1
        if args.cuda:
            I, dx, dy, feature, recon  = I.cuda(), dx.cuda(), dy.cuda(), feature.cuda(), recon.cuda()
        optimizer.zero_grad()
        # dx & dy
        grad_input = torch.cat([dx, dy], 1)
        nI = network(I, feature, grad_input)
        grad = grad_generator.predict(torch.cat([I, feature, grad_input], 1))
        # grad = None
        L1_loss, Grad_loss, Feature_loss = calc_loss(nI, I, dx, dy, feature, grad=grad , recon=recon)
        loss = L1_loss + Grad_loss + Feature_loss
        train_loss['L1_loss'] += L1_loss.item()
        train_loss['Grad_loss'] += Grad_loss.item()
        train_loss['Feature_loss'] += Feature_loss.item()
        loss.backward()
        optimizer.step()
        # i, dx, dy, feature, recon = dataloader.next()
        print('step %d/%d total-loss: %f l1: %f grad: %f feature: %f %fs'%(step, len(train_loader), loss.item(), L1_loss.item(), Grad_loss.item(), Feature_loss.item(), time.time() - start_time))
    # average over batches
    for key in train_loss.keys():
        train_loss[key] /= len(train_loader)
    all_loss = train_loss['L1_loss'] + train_loss['Grad_loss'] + train_loss['Feature_loss']
    writer.add_scalar('l1 loss', train_loss['L1_loss'],epoch)
    writer.add_scalar('all loss', all_loss,epoch)
    writer.add_scalar('Grad loss', train_loss['Grad_loss'],epoch)
    writer.add_scalar('Feature loss', train_loss['Feature_loss'],epoch)

    print('Training --- L1_loss: {L1_loss:.6f}, Grad_loss: {Grad_loss:.6f}, Feature_loss: {Feature_loss:.6f}'.format(**train_loss))




def test(network,test_loader, epoch):
    network.eval()
    grad_generator.set_training(False)
    test_loss = {'L1_loss': 0.0, 'Grad_loss': 0.0, 'Feature_loss': 0.0}
    step = 0
    with torch.no_grad():
        for I, dx, dy, feature, recon in test_loader:
            step += 1
            if args.cuda:
                I, dx, dy, feature, recon = I.cuda(), dx.cuda(), dy.cuda(), feature.cuda(), recon.cuda()
            # image & feature
            # dx & dy
            grad_input = torch.cat([dx, dy], 1)
            nI = network(I, feature, grad_input)
            grad = grad_generator.predict(torch.cat([I, feature, grad_input], 1))
            # grad = None
            L1_loss, Grad_loss, Feature_loss = calc_loss(nI, I, dx, dy, feature, grad=grad, recon=recon)
            loss = L1_loss + Grad_loss + Feature_loss
            test_loss['L1_loss'] += L1_loss.item()
            test_loss['Grad_loss'] += Grad_loss.item()
            test_loss['Feature_loss'] += Feature_loss.item()

    # average over batches
    for key in test_loss.keys():
        test_loss[key] /= len(test_loader)
    writer.add_scalar('test-l1 loss', test_loss['L1_loss'], epoch)
    writer.add_scalar('test-all loss', loss.item(), epoch)
    writer.add_scalar('test-Grad loss', test_loss['Grad_loss'], epoch)
    writer.add_scalar('test-Feature loss', test_loss['Feature_loss'], epoch)
    print('Testing --- L1_loss: {L1_loss:.6f}, Grad_loss: {Grad_loss:.6f}, Feature_loss: {Feature_loss:.6f}'.format(**test_loss))

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Vol-GradNet')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--process_count', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--lambda_L1', type=float, default=1)
    parser.add_argument('--lambda_F', type=float, default=2)
    parser.add_argument('--mu', type=float, default=16.0)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default = 202015219)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--feature_channel', type=int, default=7)
    args = parser.parse_args()

    # init cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set random seed
    torch.manual_seed(args.seed)

    # make log and result dir
    if not os.path.exists('logs/'+save_dir+ save_name):
        os.makedirs('logs/'+save_dir+ save_name)
    if not os.path.exists('results/'+save_dir+ save_name):
        os.makedirs('results/'+save_dir+ save_name)

    # init network
    network = VolGradNet(feature_channel=args.feature_channel)
    loss_func = LossFunction(args)
    grad_generator = GradGenerator(args, loss_func, feature_channel=args.feature_channel)


    # check cuda
    if args.cuda:
        print('using cuda')
        torch.cuda.manual_seed(args.seed)
        network = torch.nn.DataParallel(network)
        network.cuda()
    else:
        print('using CPU')

    train_loader = torch.utils.data.DataLoader(Dataset(training=True, dir='data/', feature_num=args.feature_channel),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.process_count)
    test_loader = torch.utils.data.DataLoader(Dataset(training=False, dir='data/', feature_num=args.feature_channel),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.process_count)

    # optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, betas=(args.beta, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.95)

    # train & test
    temp_lambda_F = args.lambda_F
    args.lambda_F = 0

    for epoch in range(1, args.epochs):
        print('Epoch %d' % (epoch ))

        # start time
        print('train data branch epoch %d' % epoch)
        epoch_start_time = time.time()
        train(network, optimizer, train_loader, epoch)
        print('cost: %f s'%(time.time() - epoch_start_time))
        print('--------------------')

        print('train gradient generator epoch %d' % epoch)
        epoch_start_time = time.time()
        grad_generator.unfreeze()
        train_grad_generator(train_loader, epoch)
        grad_generator.freeze()
        print('cost: %f s' % (time.time() - epoch_start_time))
        print('--------------------')
        grad_generator.scheduler_step()


        print('test epoch %d' % epoch)
        epoch_start_time = time.time()
        test(network, test_loader, epoch)
        print('cost: %f s' % (time.time() - epoch_start_time))
        print('--------------------')

        scheduler.step()
        if epoch % args.save_interval == 0:
            torch.save(network.state_dict(), 'results/%s%s/model_%d.pkl' % (save_dir,save_name, epoch))
            grad_generator.save('results/%s%s/g_model_%d.pkl' % (save_dir,save_name, epoch))
        if epoch >= 5 :
            if args.lambda_F == 0.0:
                args.lambda_F = 0.1
            else:
                args.lambda_F = min(args.lambda_F * 1.05, temp_lambda_F)

