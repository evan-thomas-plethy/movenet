from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import json
import os
from pathlib import Path

from models.model import create_model, load_model
from utils.image import square_padding_resize
from utils.image import get_affine_transform, affine_transform
from utils.debugger import Debugger


class BaseDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads,
                                  opt.head_conv, opt.froze_backbone)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()
        
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.opt = opt
        self.pause = True
        self.global_num = 0

    def pre_process(self, image, meta=None):
        height, width = image.shape[0:2]

        new_height = 256
        new_width = 256
        target_size = (new_width, new_height)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.opt.preserve_aspect_ratio:
            inp_image, original_dims = square_padding_resize(image, target_size)
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([new_width, new_height], dtype=np.float32)
        else:
            c = np.array([width / 2., height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
            rot = 0
            trans_input = get_affine_transform(
                c, s, rot, [new_width, new_height])
            inp_image = cv2.warpAffine(image, trans_input,
                                      (new_width, new_height),
                                      flags=cv2.INTER_LINEAR)

        inp_image = (inp_image.astype(np.float32) / 127.5)
        inp_image = (inp_image - self.mean) / self.std
        inp_image = inp_image.transpose(2, 0, 1)
        
        images = inp_image.reshape(1, 3, new_height, new_width)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'in_height': height,
                'in_width': width,
                'out_height': new_height // self.opt.down_ratio,
                'out_width': new_width // self.opt.down_ratio}
        return images, meta

    def process(self, images, return_time=False):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1):
        raise NotImplementedError

    def show_results(self, debugger, image, results):
        raise NotImplementedError

    def run(self, image_or_path_or_tensor, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                            theme=self.opt.debugger_theme)
        start_time = time.time()
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        # detections = []
        scale_start_time = time.time()

        images, meta = self.pre_process(image, meta)
        images = images.to(self.opt.device)
        # torch.cuda.synchronize()
        pre_process_time = time.time()
        pre_time += pre_process_time - scale_start_time

        # if "ANKLE_DORSIFLEX_SITTING_3_frames0012" in image_or_path_or_tensor:
        #     pred_dir = '/home/ubuntu/visionAI/movenet/images/val_pipeline_inputs'
        #     os.makedirs(pred_dir, exist_ok=True)
        #     pred_path = os.path.join(pred_dir, f'{image_or_path_or_tensor.split("/")[-1].split(".")[0]}.json')
        #     with open(pred_path, 'w') as f:
        #         json.dump(images.tolist(), f)
        #     print(f"Saved input to {pred_path}")

        output, dets, forward_time = self.process(images, return_time=True, image_path=image_or_path_or_tensor)

        if "ANKLE_DORSIFLEX_SITTING_3_frames0002" in image_or_path_or_tensor:
            pred_dir = '/home/ubuntu/visionAI/movenet/images/val_pipeline_dets'
            os.makedirs(pred_dir, exist_ok=True)
            pred_path = os.path.join(pred_dir, f'{image_or_path_or_tensor.split("/")[-1].split(".")[0]}.json')
            with open(pred_path, 'w') as f:
                json.dump(dets.tolist(), f)
            print(f"Saved dets to {pred_path}")

        # torch.cuda.synchronize()
        net_time += forward_time - pre_process_time
        decode_time = time.time()
        dec_time += decode_time - forward_time
        if self.opt.debug >= 2:
            self.debug(debugger, images, dets, output)
        dets = self.post_process(dets, meta)

        if "ANKLE_DORSIFLEX_SITTING_3_frames0002" in image_or_path_or_tensor:
            pred_dir = '/home/ubuntu/visionAI/movenet/images/val_pipeline_dets_post_process'
            os.makedirs(pred_dir, exist_ok=True)
            pred_path = os.path.join(pred_dir, f'{image_or_path_or_tensor.split("/")[-1].split(".")[0]}.json')
            with open(pred_path, 'w') as f:
                json.dump(dets.tolist(), f)
            print(f"Saved dets to {pred_path}")

        # torch.cuda.synchronize()
        post_process_time = time.time()
        post_time += post_process_time - decode_time
        results = dets

        # ---------- SAVE TO JSON ----------
        # Try to get a filename based on input image
        filename = None
        if isinstance(image_or_path_or_tensor, str):
            filename = Path(image_or_path_or_tensor).stem
        else:
            # fallback: use a timestamp or unique ID if input was a tensor
            filename = f"result_{int(time.time()*1000)}"

        # Prepare output directory
        save_dir = getattr(self.opt, 'save_dir', './outputs')
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Save results to JSON
        save_path = os.path.join(save_dir, f'{filename}.json')
        with open(save_path, 'w') as f:
            json.dump(results.tolist(), f, indent=2)
        # ---------- DONE SAVING TO JSON ----------

        # results = self.merge_outputs(detections)
        # torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        if self.opt.debug >= 1:
            results_copy = results.copy()
            results_copy[:, [0, 1]] = results_copy[:, [1, 0]]
            self.show_results(debugger, image, results_copy, prefix=self.global_num)
            self.global_num += 1

        return {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time}
