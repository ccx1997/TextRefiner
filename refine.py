import os
import argparse
import torch
import cv2
from model.seq2seq import Encoder, AttentionDecoder, RefineText
from torchvision import transforms
from data.load_data import get_alphabet, get_dataloader
from model_utils import Ids2Str, batch_test
from tensorboardX import SummaryWriter


def decorate_model(model, is_training, device):
    model = model.to(device)
    if is_training:
        model.train()
    else:
        model.eval()


def load_vanilla(num_classes, device, args):
    encoder = Encoder(use_stn=args.use_stn).to(device)
    decoder = AttentionDecoder(hidden_dim=256, attention_dim=256, y_dim=num_classes, encoder_output_dim=512,
                               f_lookup_ts="/home/dataset/TR/synth_cn/lookup.pt").to(device)  # y_dim for classes_num
    decorate_model(encoder, is_training=False, device=device)
    decorate_model(decoder, is_training=False, device=device)
    checkpoint = torch.load(args.pre_ocr)
    encoder.load_state_dict(checkpoint["state_dict"]["encoder"])
    decoder.load_state_dict(checkpoint["state_dict"]["decoder"], strict=False)
    return encoder, decoder


def get_loader(is_training, alphabet, bs):
    lmdb_dir = "/home/dataset/TR/synth_cn/train_lmdb/" if is_training else "/home/dataset/TR/synth_cn/test_lmdb/"
    input_size = [64, 256] if args.use_stn else [32, 128]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trsf = transforms.Compose([transforms.ToTensor(), normalize])
    dataloader = get_dataloader(lmdb_dir, alphabet, input_size, trsf, batch_size=bs, is_train=is_training)
    return dataloader


def once_vanilla(img_name):
    image_np = cv2.imread(img_name, 1)
    transf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    input_size = (256, 64) if args.use_stn else (128, 32)
    image = transf(cv2.resize(image_np, input_size))
    image = image.unsqueeze(0).to(device)  # [1, 3, h, w]
    # forward propagation
    (predicted, _, cp, cf), _ = batch_test(
        image, encoder, decoder, max_len=40, need_refine=True)  # [1, L], [1, L], [1, L, c]
    return predicted, cp, cf


def train_refiner(num_classes, encoder, decoder, alphabet, lr, epoch, step, device, label_map, model_name, pre_train=""):
    refiner = RefineText(num_classes=num_classes, feat_dim=512, embedding_dim=200,
                         f_lookup_ts="/home/dataset/TR/synth_cn/lookup.pt")
    decorate_model(refiner, is_training=True, device=device)
    if pre_train:
        print('Use pre-trained model -- ' + pre_train)
        try:
            refiner.load_state_dict(torch.load(pre_train))
        except:
            print("load parameters using strict=False")
            refiner.load_state_dict(torch.load(pre_train), strict=False)
    train_loader = get_loader(is_training=True, alphabet=alphabet, bs=32)
    # optimizer = torch.optim.Adadelta(refiner.parameters(), lr=lr)
    optimizer = torch.optim.Adam(refiner.parameters(), lr=lr, weight_decay=1e-4)
    lr_step = [50000, 100000, 200000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_step, gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    total_loss, best_loss = 0.0, 1000.0
    tfLogger = SummaryWriter('./log_ref')
    while True:
        for batch_idx, (imgs, (targets, targets_len), idx) in enumerate(train_loader):
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            imgs, targets, targets_len = imgs.to(device), targets.to(device), targets_len.to(device)
            if epoch >= 5:
                (predicted0, _, cp, cf), _ = batch_test(
                    imgs, encoder, decoder, max_len=40, need_refine=True)  # [b, L+1], [b, L+1], [b, L+1, c]
                cf.detach_()
                cp.detach_()
            else:
                # For the early training, we consider only the LM
                predicted0 = targets.permute(1, 0)
                cp = torch.ones_like(predicted0).cuda().float() * 0.5
                cf = torch.randn(predicted0.size(0), predicted0.size(1), 512).cuda()
            targets = targets.permute(1, 0)[:, :-1]    # [b, L]
            y = refiner(cf[:, :targets.size(1), :], cp[:, :targets.size(1)], targets)    # [b, nc, L]
            # loss = loss_fn(y, targets)
            loss_vis = loss_fn(y[0], targets)
            loss_sem = loss_fn(y[1], targets)
            loss_fusion = loss_fn(y[2], targets)
            loss = loss_vis * 0.2 + loss_sem * 0.1 + loss_fusion
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            step += 1
            if step % 500 == 0:
                pred_string0 = label_map.decode(predicted0[:, :-1].squeeze(0).cpu().numpy())
                _, predicted1 = torch.max(y[2], dim=1)
                pred_string1 = label_map.decode(predicted1.squeeze(0).cpu().numpy())
                gt_string = label_map.decode(targets.squeeze(0).cpu().numpy())
                print('==pred_bef:{0}\n==pred_now:{1}\n========gt:{2}'.format(pred_string0, pred_string1, gt_string))
                print("loss_vis:{}  loss_sem:{}  loss_fusion:{}  loss_total:{}\n".format(
                    loss_vis.item(), loss_sem.item(), loss_fusion.item(), loss.item()))
            if step % 2000 == 0:
                avg_loss = total_loss / 2000.
                total_loss = 0
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    print('saved in ' + model_name)
                    torch.save(refiner.state_dict(), model_name)
                print("Epoch: {0}  Step: {1:.1f}k  Loss: {2:.4f}  BestLoss: {3:.4f}".format(epoch, step/1000, avg_loss, best_loss))

                info = {'loss': avg_loss, 'learning_rate': lr}
                for tag, value in info.items():
                    tfLogger.add_scalar(tag, value, step)

            # if step % 100000 == 0:
        epoch += 1
    tfLogger.close()


def test_refiner(num_classes, encoder, decoder, alphabet, device, label_map, pre_train=""):
    refiner = RefineText(num_classes=num_classes, feat_dim=512, embedding_dim=200,
                         f_lookup_ts="/home/dataset/TR/synth_cn/lookup.pt")
    decorate_model(refiner, is_training=False, device=device)
    print('Use pre-trained model -- ' + pre_train)
    refiner.load_state_dict(torch.load(pre_train))
    test_loader = get_loader(is_training=False, alphabet=alphabet, bs=1)
    f_results = open('pred_refiner.txt', 'w')
    cnt_former = 0
    cnt_true = 0
    for batch_idx, (imgs, (targets, targets_len), idx) in enumerate(test_loader):
        imgs, targets, targets_len = imgs.to(device), targets.to(device), targets_len.to(device)
        targets = targets.permute(1, 0)[:, :-1]    # [b, L]
        (predicted0, _, cp, cf), _ = batch_test(
            imgs, encoder, decoder, max_len=40, need_refine=True)  # [b, L+1], [b, L+1], [b, L+1, c]
        # if not targets.equal(predicted0[:, :-1]):
        #     print("checking...")
        predicted1 = refiner(cf[:, :-1, :], cp[:, :-1], predicted0[:, :-1])    # [b, L]
        pred_string0 = label_map.decode(predicted0[:, :-1].squeeze(0).cpu().numpy())
        pred_string1 = label_map.decode(predicted1.squeeze(0).cpu().numpy())
        gt_string = label_map.decode(targets.squeeze(0).cpu().numpy())
        if pred_string0 == gt_string:
            cnt_former += 1
        if pred_string1 == gt_string:
            cnt_true += 1
        record = 'pred_before: {0}  pred_now: {1}  gt: {2}'.format(pred_string0, pred_string1, gt_string)
        f_results.writelines(record+'\n')
        if batch_idx % 1000 == 999:
        # if pred_string0 != gt_string:
            print(record)
            # print(cp[:, :-1])
    f_results.close()
    print("Accr={}/{}={:.2%}, previous_accr={}/{}={:.2%}".format(
        cnt_true, batch_idx+1, cnt_true/(batch_idx+1), cnt_former, batch_idx+1, cnt_former/(batch_idx+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--idc", type=str, default="1")
    parser.add_argument('--pre_ocr', type=str, default='logs/model-520000.pth')
    parser.add_argument('--param', type=str, default='')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate (default: 1.0)')
    parser.add_argument('--epoch', type=int, default=0, help='from x-th epoch to start')
    parser.add_argument('--step', type=int, default=1, help='from x-th step to start')
    parser.add_argument('--save_name', type=str, default='./param/refiner_param.pkl')
    parser.add_argument('--use_stn', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.idc
    device = torch.device('cuda:0')

    if not os.path.exists('./param'):
        os.mkdir('param')

    f_alphabet = "/home/dataset/TR/synth_cn/alphabet.json"
    alphabet = get_alphabet(f_alphabet)
    label_map = Ids2Str(alphabet)
    num_classes = label_map.num_classes

    # get model
    encoder, decoder = load_vanilla(num_classes, device, args)

    if args.test:
        test_refiner(num_classes, encoder, decoder, alphabet, device, label_map, pre_train=args.param)
    else:
        train_refiner(num_classes, encoder, decoder, alphabet, lr=args.lr, epoch=args.epoch, step=args.step,
                      device=device, label_map=label_map, model_name=args.save_name, pre_train=args.param)
