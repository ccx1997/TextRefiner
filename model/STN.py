from torch import nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
from model.TPS import TPSGridGen

def make_layers(cfg):
    layers = []
    in_channel = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            conv2d = nn.Conv2d(in_channel,v,kernel_size=3,padding=1)
            layers += [conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
            in_channel = v
    return nn.Sequential(*layers)


class LocNet(nn.Module):
    def __init__(self,num_control_points,activation,target_control_points):
        super(LocNet,self).__init__()

        self.activation = activation
        self.num_control_points = num_control_points

        self.conv = make_layers([32,'M',64,'M',128,'M',256,'M',256,'M',256])
        self.fc1 = nn.Sequential(
            nn.Linear(4096,512),
            nn.BatchNorm1d(512),
            nn.ReLU(True))
        self.fc2 = nn.Linear(512,self.num_control_points * 2) # [k*2]

        if self.activation == 'tanh':
            bais = torch.from_numpy(np.arctanh(target_control_points.numpy()))
            bais = bais.view(-1)
            self.fc2.bias.data.copy_(bais)
        elif self.activation == 'none':
            bais = target_control_points.view(-1)
            self.fc2.bias.data.copy_(bais)
        self.fc2.weight.data.zero_()

    def forward(self, x):
        N = x.size(0)
        x = self.conv(x)
        x = x.view(N,-1)
        x = self.fc1(x)
        if self.activation == 'tanh':
            x = F.tanh(self.fc2(x))
        elif self.activation == 'none':
            x = self.fc2(x)
        else:
            raise ValueError('Unknown activation:{}'.format(self.activation))
        ctrl_pts = x.view(N,self.num_control_points,2)
        return ctrl_pts


class STN(nn.Module):
    def __init__(self,localization_img_size=None,output_img_size=None,num_control_points=20,activation='none',margins=None):
        super(STN,self).__init__()

        self.localization_img_size = localization_img_size
        self.output_img_size = output_img_size
        self.num_control_points = num_control_points
        self.activation = activation
        self.margins = margins
        self.target_control_points = self._build_target_control_points(self.margins)

        self.loc_net = LocNet(self.num_control_points,self.activation,self.target_control_points)
        print(self.loc_net)
        self.tps = TPSGridGen(self.output_img_size,self.target_control_points)


    def forward(self,x,source_control_points=None):
        # import ipdb
        # ipdb.set_trace()
        batch_size = x.size(0)
        if  source_control_points is None:
            source_control_points = self.loc_net(x)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size,self.output_img_size[0],self.output_img_size[1],2)
        rectified_img = self.grid_sample(x,grid)

        return rectified_img #,source_coordinate



    def _build_target_control_points(self,margins):
        margin_x, margin_y = margins
        num_ctrl_pts_per_side = self.num_control_points // 2
        ctrl_pts_x = np.linspace(margin_x - 1.0, 1.0 - margin_x, num_ctrl_pts_per_side)
        ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * (margin_y-1.0)
        ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        output_ctrl_pts = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return torch.Tensor(output_ctrl_pts)

    def grid_sample(self,img,batch_grid,canvas=None):
        assert img.dtype == torch.float32
        output = F.grid_sample(img,batch_grid)
        if not canvas:
            return output
        else:
            input_mask = Variable(img.data.new(img.size()).fill_(1))
            output_mask = F.grid_sample((input_mask,batch_grid))
            padded_output = output * output_mask + canvas * (1-output_mask)
            return padded_output
