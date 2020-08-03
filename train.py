import time, os, datetime
import argparse, sys
import numpy as np
import torch
from torch import optim
from utils.utils import Logger, save_state, load_state
from data.load_data import get_dataloader, get_alphabet
from torchvision import transforms
from model.seq2seq import Encoder, AttentionDecoder
from model_utils import batch_train, batch_test, Ids2Str
from loss import SequenceCrossEntropyLoss
from tensorboardX import SummaryWriter

f_alphabet = "/home/dataset/TR/synth_cn/alphabet.json"
alphabet = get_alphabet(f_alphabet)
converter = Ids2Str(alphabet)


def test(encoder, decoder, test_loader, step=1, tfLogger=None):
    total, correct = 0, 0
    start = time.time()
    encoder.eval(), decoder.eval()
    cnt_batch = 0
    for batch_idx, (imgs, (targets, lengths), _) in enumerate(test_loader):
        total += imgs.size(0)
        input_tensor = imgs.cuda()
        preds, _ = batch_test(input_tensor, encoder, decoder)  # [b,t]
        targets = targets.numpy()  # [t,b]
        pred_seq = converter.decode(preds.cpu().numpy())
        target_seq = converter.decode(np.transpose(targets))
        for label, pred in zip(target_seq, pred_seq):
            if cnt_batch == 0:
                print('===' * 10)
                print(' pred: %s' % pred)
                print('label: %s' % label)
            if label == pred:
                correct += 1
        cnt_batch += 1
    print('Finished testing in {:.2f}s\tAccuracy:{:.2f}%'.format(time.time() - start, 100. * (correct / total)))

    if tfLogger is not None:
        info = {
            'accuracy': correct / total,
        }
        for tag, value in info.items():
            tfLogger.add_scalar(tag, value, step)
    return correct / total


def main(args):
    input_size = [64, 256] if args.use_stn else [32, 128]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_trsf = transforms.Compose([transforms.ToTensor(), normalize])
    train_loader = get_dataloader(args.training_data, alphabet, input_size, train_trsf, args.batch_size)
    test_trsf = transforms.Compose([transforms.ToTensor(), normalize])
    test_loader = get_dataloader(args.test_data, alphabet, input_size, test_trsf, args.batch_size, is_train=False)

    encoder = Encoder(use_stn=args.use_stn).cuda()
    decoder = AttentionDecoder(hidden_dim=256, attention_dim=256, y_dim=converter.num_classes, encoder_output_dim=512,
                               f_lookup_ts="/home/dataset/TR/synth_cn/lookup.pt").cuda()  # output_size for classes_num
    encoder_optimizer = optim.Adadelta(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.Adadelta(decoder.parameters(), lr=args.lr)
    optimizers = [encoder_optimizer, decoder_optimizer]
    lr_step = [100000, 300000, 500000]
    # lr_step = [100000, 200000]
    encoder_scheduler = optim.lr_scheduler.MultiStepLR(encoder_optimizer, lr_step, gamma=0.1)
    decoder_scheduler = optim.lr_scheduler.MultiStepLR(decoder_optimizer, lr_step, gamma=0.1)
    criterion = SequenceCrossEntropyLoss()

    step, total_loss, best_res = 1, 0, 0
    # For fine-tuning
    # checkpoint = torch.load('./logs/model-120000.pth')
    # encoder.load_state_dict(checkpoint["state_dict"]["encoder"])
    # decoder.load_state_dict(checkpoint["state_dict"]["decoder"], strict=False)

    if args.restore_step > 0:
        step = args.restore_step
        load_state(args.logs_dir, step, encoder, decoder, optimizers)

    sys.stdout = Logger(os.path.join(args.logs_dir, 'log.txt'))
    train_tfLogger = SummaryWriter(os.path.join(args.logs_dir, 'train'))
    test_tfLogger = SummaryWriter(os.path.join(args.logs_dir, 'test'))

    # start training
    while True:
        for batch_idx, (imgs, (targets, targets_len), idx) in enumerate(train_loader):

            input_data, targets, targets_len = imgs.cuda(), targets.cuda(), targets_len.cuda()
            encoder_optimizer.zero_grad(), decoder_optimizer.zero_grad()

            loss, recitified_img = batch_train(input_data, targets, targets_len, encoder, decoder, criterion, 1.0)
            encoder_optimizer.step(), decoder_optimizer.step()
            encoder_scheduler.step(), decoder_scheduler.step()
            total_loss += loss

            if step % 500 == 0:
                print('==' * 30)
                preds, _ = batch_test(input_data, encoder, decoder)
                print('preds: ', converter.decode(preds.cpu().numpy()))
                print('==' * 30)
                print('label: ', converter.decode(targets.permute(1, 0).cpu().numpy()))
                encoder.train(), decoder.train()

            if step % args.log_interval == 0:
                print('{} step:{}\tLoss: {:.6f}'.format(
                    datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                    step, total_loss / args.log_interval))
                if train_tfLogger is not None:
                    """
                    x = vutils.make_grid(input_data.cpu())
                    train_tfLogger.add_image('train/input_img', x, step)
                    if args.use_stn:
                        x = vutils.make_grid(recitified_img.cpu())
                        train_tfLogger.add_image('train/recitified_img', x, step)
                    """
                    for param_group in encoder_optimizer.param_groups:
                        lr = param_group['lr']
                    info = {'loss': total_loss / args.log_interval,
                            'learning_rate': lr}
                    for tag, value in info.items():
                        train_tfLogger.add_scalar(tag, value, step)
                total_loss = 0
            if step % args.save_interval == 0:
                # save params
                save_state(args.logs_dir, step, encoder, decoder, optimizers)

                # Test after an args.save_interval
                res = test(encoder, decoder, test_loader, step=step, tfLogger=test_tfLogger)
                is_best = res >= best_res
                best_res = max(res, best_res)
                print('\nFinished step {:3d}  TestAcc: {:.4f}  best: {:.2%}{}\n'.
                      format(step, res, best_res, ' *' if is_best else ''))
                encoder.train(), decoder.train()

            step += 1

    # Close the tf logger
    train_tfLogger.close()
    test_tfLogger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ASTER')
    parser.add_argument('--idc', type=str, default='1', help="Choose the id of gpu")
    parser.add_argument('--training_data', type=str,
                        default="/home/dataset/TR/synth_cn/train_lmdb/")
    parser.add_argument('--test_data', type=str, default="/home/dataset/TR/synth_cn/test_lmdb/")
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--save_interval', type=int, default=10000, metavar='N',
                        help='how many steps to wait before saving model parmas')
    parser.add_argument('--log_interval', type=int, default=500, metavar='N',
                        help='how many steps to wait before print periodically')
    parser.add_argument('--logs_dir', type=str, metavar='PATH', default='./logs')
    parser.add_argument('--restore_step', type=int, default=0, help='restore for restore_step')
    parser.add_argument('--use_stn', action='store_true', default=False)
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.idc
    main(opt)
