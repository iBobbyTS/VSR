import math
import numpy
import torch

from basicsr.models.archs.edvr_arch import EDVR


class SRer:
    models = {
        'ld': [EDVR(num_feat=128, num_reconstruct_block=40, hr_in=True, with_predeblur=True).cuda(), 5, 1],
        'ldc': [EDVR(num_feat=128, num_reconstruct_block=40, hr_in=True, with_predeblur=True).cuda(), 5, 1],
        'l4r': [EDVR(num_feat=128, num_reconstruct_block=40).cuda(), 5, 4],
        'l4v': [EDVR(num_feat=128, num_frame=7, num_reconstruct_block=40).cuda(), 7, 4],
        'l4br': [EDVR(num_feat=128, num_reconstruct_block=40, with_predeblur=True).cuda(), 5, 4],
        'm4r': [EDVR(with_tsa=False).cuda(), 5, 4],
        'mt4r': [EDVR().cuda(), 5, 4]
    }

    def __init__(self, model_name, model_path, height, width):
        self.model, self.num_frame, self.enlarge = self.models[model_name]

        self.h_w = [int(math.ceil(height / 32) * 32 - height) if height % 32 else 0,
                    int(math.ceil(width / 32) * 32) - width if width % 32 else 0]
        dim = [height + self.h_w[0], width + self.h_w[1]]

        self.batch = torch.cuda.FloatTensor(1, self.num_frame, 3, dim[0], dim[1])

        self.model.load_state_dict(torch.load(model_path)['params'], strict=True)
        self.model.eval()

    def ndarray2tensor(self, frames: list):  # 内部调用
        out_frames = []
        for frame in frames:
            frame = torch.cuda.ByteTensor(frame)[:, :, [2, 1, 0]]
            frame = torch.cat([torch.zeros((frame.shape[0], self.h_w[1], 3)).cuda().byte(), frame], dim=1)
            frame = torch.cat([torch.zeros((self.h_w[0], frame.shape[1], 3)).cuda().byte(), frame], dim=0)
            out_frames.append(frame.permute(2, 0, 1).float() / 255)
        return out_frames

    def sr(self):
        output = self.model(self.batch).data.squeeze().float() * 255
        output = output.clamp(0, 255).byte().permute(1, 2, 0)[self.h_w[0]*self.enlarge:, self.h_w[1]*self.enlarge:, [2, 1, 0]].cpu().numpy()
        return output
