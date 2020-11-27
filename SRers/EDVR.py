import math

import numpy
import torch

from basicsr.models.archs.edvr_arch import EDVR


def cal_split(h_w, max_side_length):
    splition = []
    for i in range(math.ceil(h_w[0]/max_side_length)):
        for j in range(math.ceil(h_w[1] / max_side_length)):
            ws = i*max_side_length
            we = (i+1)*max_side_length if (i+1)*max_side_length <= h_w[0] else h_w[0]
            hs = j*max_side_length
            he = (j+1)*max_side_length if (j+1)*max_side_length <= h_w[1] else h_w[1]
            splition.append((ws, we, hs, he))
    return splition


class SRer:
    models = {
        'ld':   [EDVR(num_feat=128, num_reconstruct_block=40, hr_in=True, with_predeblur=True).cuda(), 5, 1],
        'ldc':  [EDVR(num_feat=128, num_reconstruct_block=40, hr_in=True, with_predeblur=True).cuda(), 5, 1],
        'l4r':  [EDVR(num_feat=128, num_reconstruct_block=40).cuda(), 5, 4],
        'l4v':  [EDVR(num_feat=128, num_reconstruct_block=40, num_frame=7, center_frame_idx=3).cuda(), 7, 4],
        'l4br': [EDVR(num_feat=128, num_reconstruct_block=40, with_predeblur=True).cuda(), 5, 4],
        'm4r':  [EDVR(with_tsa=False).cuda(), 5, 4],
        'mt4r': [EDVR().cuda(), 5, 4]
    }

    def __init__(self, model_name, model_path, height, width):
        self.model, self.num_frame, self.enlarge = self.models[model_name]
        self.height, self.width = height, width

        self.h_w = [int(math.ceil(height / 32) * 32 - height) if height % 32 else 0,
                    int(math.ceil(width / 32) * 32) - width if width % 32 else 0]
        self.dim = [height + self.h_w[0], width + self.h_w[1]]
        self.pader = torch.nn.ReplicationPad2d([0, self.h_w[1], 0, self.h_w[0]])

        self.model.load_state_dict(torch.load(model_path)['params'], strict=True)
        self.model.eval()

    def init_batch(self, video):
        self.batch = torch.cuda.FloatTensor(1, self.num_frame, 3, self.dim[0], self.dim[1])
        print(self.batch.size())
        self.splition = cal_split((self.dim[0], self.dim[1]), 256)
        frame = self.ndarray2tensor(video.read()[1])
        for i in range(self.num_frame // 2 + 1):
            self.batch[0, i] = frame
        for i in range(self.num_frame // 2+1, self.num_frame - 1):
            self.batch[0, i] = self.ndarray2tensor(video.read()[1])

    def ndarray2tensor(self, frame: numpy.ndarray):
        out_frame = torch.cuda.ByteTensor(frame)[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)
        out_frame = self.pader(out_frame.float())/255.0
        return out_frame

    def sr(self, f):
        if f[0]:
            self.batch[0, -1] = self.ndarray2tensor(f[1])
        outputs = []
        for i in self.splition:
            outputs.append((self.model(self.batch[:, :, :, i[0]:i[1], i[2]:i[3]])*255.0).clamp(0, 255).byte().squeeze().permute(1, 2, 0).cpu().numpy())
        self.batch[0, :-1] = self.batch.clone()[0, 1:]
        output = numpy.zeros((self.height*self.enlarge, self.width*self.enlarge, 3), dtype=numpy.uint8)
        for slice_, tensor in zip(self.splition, outputs):
            output[slice_[0]*self.enlarge:slice_[1]*self.enlarge, slice_[2]*self.enlarge:slice_[3]*self.enlarge, :] = tensor
        output = output[:, :, [2, 1, 0]]
        return output
