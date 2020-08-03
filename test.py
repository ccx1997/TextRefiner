import os
import argparse
import torch
import cv2
import numpy as np
from model.seq2seq import Encoder, AttentionDecoder
from torchvision import transforms
from data.load_data import get_alphabet
from model_utils import Ids2Str, batch_test, visualize_attention


def decorate_model(model):
    model = model.to(device)
    model.eval()


def one_pass(img_name):
    image_np = cv2.imread(img_name, 1)
    transf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    input_size = (256, 64) if args.use_stn else (128, 32)
    # h, w = image_np.shape[:2]
    # newh = 32
    # neww = int(newh * w / h)
    # input_size = (neww, newh)
    image = transf(cv2.resize(image_np, input_size))
    image = image.unsqueeze(0).to(device)  # [1, 3, newh, neww]

    # forward propagation
    (predicted, attention, prob, _), _ = batch_test(image, encoder, decoder,
                                                    max_len=40, visualize=True)  # [1, L], [1, L, w]
    predicted.squeeze_(0)
    attention.squeeze_(0)
    prob.squeeze_(0)
    prob = prob[:-1]
    #
    ids2str = Ids2Str(alphabet)
    pred_string = ids2str.decode(predicted.cpu().numpy())
    #
    w = image_np.shape[1]
    attention_vis = visualize_attention(attention.cpu().numpy(), w, dh=8)
    image_vis = joint_img_att(red(prob.cpu().numpy(), w), image_np, attention_vis)
    show_it(img_name, image_vis, pred_string, prob.cpu().numpy())


def show_it(img_name, img_np, pred_string, prob):
    print("The text in {0} is: {1}".format(img_name, pred_string))
    np.set_printoptions(precision=2)
    print(prob)
    cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
    cv2.imshow(img_name, img_np)
    cv2.waitKey()
    cv2.destroyAllWindows()


def red(prob, w):
    dh = 16
    prob = np.tile(prob, (1, dh)).reshape(dh, -1)
    prob = cv2.resize(prob, (w, dh), interpolation=cv2.INTER_NEAREST)
    prob = prob * 255
    tmp = np.zeros((dh, w, 3))
    tmp[:, :, 2] = prob
    return tmp.astype(np.uint8)


def joint_img_att(prob_np, img_np, att_np, rgb=True):
    att_np = att_np * 255
    att_np = att_np.astype(np.uint8)
    if rgb:
        att_np = cv2.cvtColor(att_np, cv2.COLOR_GRAY2BGR)
    return np.concatenate((prob_np, img_np, att_np), axis=0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--idc", type=str, default="1")
    parser.add_argument('--img', type=str, default='', help='path name of the image to be tested')
    parser.add_argument('--param', type=str, default='logs/model-520000.pth')
    parser.add_argument('--batch', type=int, default=8000, help='the start id of a batch')
    parser.add_argument('--use_stn', action='store_true', default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.idc
    device = torch.device('cuda:0')

    f_alphabet = "/home/dataset/TR/synth_cn/alphabet.json"
    alphabet = get_alphabet(f_alphabet)
    label_map = Ids2Str(alphabet)

    # get model
    encoder = Encoder(use_stn=args.use_stn).cuda()
    decoder = AttentionDecoder(hidden_dim=256, attention_dim=256, y_dim=label_map.num_classes, encoder_output_dim=512,
                               f_lookup_ts="/home/dataset/TR/synth_cn/lookup.pt").cuda()  # y_dim for classes_num

    decorate_model(encoder)
    decorate_model(decoder)
    checkpoint = torch.load(args.param)
    encoder.load_state_dict(checkpoint["state_dict"]["encoder"])
    decoder.load_state_dict(checkpoint["state_dict"]["decoder"], strict=False)

    if args.img:
        img_name = args.img
        one_pass(img_name)
    else:
        for i in range(args.batch, args.batch+20):
            img_name = '/home/dataset/TR/synth_cn/test/' + str(i) + '_0.jpg'
            one_pass(img_name)
