from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from .utils import *

EPS = 1e-7


class Demo_class(nn.Module):
    def __init__(self):
        super(Demo_class, self).__init__()
        self.device = 'cuda:0'
        self.checkpoint_path = './files/checkpoint030.pth'
        self.detect_human_face = False
        self.render_video = False
        self.output_size = 256
        self.image_size = 64
        self.min_depth = 0.9
        self.max_depth = 1.1
        self.border_depth = 1.05
        self.xyz_rotation_range = 60
        self.xy_translation_range = 0.1
        self.z_translation_range = 0
        self.fov = 10  # in degrees

        self.depth_rescaler = lambda d : (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth  # (-1,1) => (min_depth,max_depth)
        self.depth_inv_rescaler = lambda d :  (d-self.min_depth) / (self.max_depth-self.min_depth)  # (min_depth,max_depth) => (0,1)


        ## NN models
        self.netD = EDDeconv(cin=3, cout=1, nf=64, zdim=256, activation=None)

        self.netD = self.netD.to(self.device)
        self.load_checkpoint()
        
        self.netD.eval()

    def load_checkpoint(self):
        print(f"Loading checkpoint from {self.checkpoint_path}")
        cp = torch.load(self.checkpoint_path)
        self.netD.load_state_dict(cp['netD'])
        # self.netA.load_state_dict(cp['netA'])
        # self.netL.load_state_dict(cp['netL'])
        # self.netV.load_state_dict(cp['netV'])

    def depth_to_3d_grid(self, depth, inv_K=None):
        if inv_K is None:
            inv_K = self.inv_K
        b, h, w = depth.shape
        grid_2d = get_grid(b, h, w, normalize=False).to(depth.device)  # Nxhxwx2
        depth = depth.unsqueeze(-1)
        grid_3d = torch.cat((grid_2d, torch.ones_like(depth)), dim=3)
        grid_3d = grid_3d.matmul(inv_K.transpose(2,1)) * depth
        return grid_3d

    def get_normal_from_depth(self, depth):
        b, h, w = depth.shape
        grid_3d = self.depth_to_3d_grid(depth)

        tu = grid_3d[:,1:-1,2:] - grid_3d[:,1:-1,:-2]
        tv = grid_3d[:,2:,1:-1] - grid_3d[:,:-2,1:-1]
        normal = tu.cross(tv, dim=3)

        zero = normal.new_tensor([0,0,1])
        normal = torch.cat([zero.repeat(b,h-2,1,1), normal, zero.repeat(b,h-2,1,1)], 2)
        normal = torch.cat([zero.repeat(b,1,w,1), normal, zero.repeat(b,1,w,1)], 1)
        normal = normal / (((normal**2).sum(3, keepdim=True))**0.5 + EPS)
        return normal

    def detect_face(self, im):
        print("Detecting face using MTCNN face detector")
        try:
            bboxes, prob = self.face_detector.detect(im)
            w0, h0, w1, h1 = bboxes[0]
        except:
            print("Could not detect faces in the image")
            return None

        hc, wc = (h0+h1)/2, (w0+w1)/2
        crop = int(((h1-h0) + (w1-w0)) /2/2 *1.1)
        im = np.pad(im, ((crop,crop),(crop,crop),(0,0)), mode='edge')  # allow cropping outside by replicating borders
        h0 = int(hc-crop+crop + crop*0.15)
        w0 = int(wc-crop+crop)
        return im[h0:h0+crop*2, w0:w0+crop*2]

    def run(self, pil_im):
        im = (pil_im)   

        _,h,w = im.shape
        im = im.unsqueeze(0)
        # resize to 128 first if too large, to avoid bilinear downsampling artifacts
        if h > self.image_size*4 and w > self.image_size*4:
            im = nn.functional.interpolate(im, (self.image_size*2, self.image_size*2), mode='bilinear', align_corners=False)
        im = nn.functional.interpolate(im, (self.image_size, self.image_size), mode='bilinear', align_corners=False)

        with torch.no_grad():
            self.input_im = (im)*2.-1.
            b, c, h, w = self.input_im.shape
            #################
            fx = (self.image_size-1)/2/(np.tan(self.fov/2 *np.pi/180))
            fy = (self.image_size-1)/2/(np.tan(self.fov/2 *np.pi/180))
            cx = (self.image_size-1)/2
            cy = (self.image_size-1)/2
            K = [[fx, 0., cx],
                [0., fy, cy],
                [0., 0., 1.]]
            K = torch.FloatTensor(K).to(self.input_im.device)
            self.inv_K = torch.inverse(K).unsqueeze(0)
            self.K = K.unsqueeze(0)
            ## predict canonical depth
            self.canon_depth_raw = self.netD(self.input_im).squeeze(1)  # BxHxW
            self.canon_depth = self.canon_depth_raw - self.canon_depth_raw.view(b,-1).mean(1).view(b,1,1)
            self.canon_depth = self.canon_depth.tanh()
            self.canon_depth = self.depth_rescaler(self.canon_depth)

            ## clamp border depth
            depth_border = torch.zeros(1,h,w-4).to(self.input_im.device)
            depth_border = nn.functional.pad(depth_border, (2,2), mode='constant', value=1)
            self.canon_depth = self.canon_depth*(1-depth_border) + depth_border *self.border_depth
            self.canon_normal = self.get_normal_from_depth(self.canon_depth)
            self.canon_depth = nn.functional.interpolate(self.canon_depth.unsqueeze(1), (self.output_size, self.output_size), mode='bilinear', align_corners=False).squeeze(1)
            self.canon_normal = nn.functional.interpolate(self.canon_normal.permute(0,3,1,2), (self.output_size, self.output_size), mode='bilinear', align_corners=False)
            self.canon_normal = torch.squeeze(self.canon_normal)
        
            if self.render_video:
                self.render_animation()
        return self.canon_depth,self.canon_normal