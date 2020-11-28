import time

everything_start_time = time.time()

import os
import json
import argparse

import cv2
import numpy

import SRers

parser = argparse.ArgumentParser()

# Input/output file
parser.add_argument('-i', '--input',  # Input file
                    type=str,
                    help='path of video to be converted')
parser.add_argument('-o', '--output',  # Output file
                    type=str, default='default',
                    help='Specify output file name. Default: output.mp4')
parser.add_argument('-ot', '--output_type',  # Output file type
                    type=str, choices=['video', 'npz', 'npy', 'tiff', 'png'], default='npy',
                    help='Output file type, -o needs to be a file and image sequence or npz needs to be a folder')
# Process type
parser.add_argument('-a', '--algorithm', type=str, default='EDVR',  # 算法
                    choices=['EDVR', 'ESRGAN'], help='EDVR or ESRGAN')
parser.add_argument('-mn', '-model_name', type=str, default='mt4r',
                    choices=['ld', 'ldc', 'l4r', 'l4v', 'l4br', 'm4r', 'mt4r'],
                    help='ld: L Deblur, ldc: L Deblur Comp, l4r: L SR REDS x4, l4v: L SR vimeo90K 4x, '
                         'l4br: L SRblur REDS 4x, m4r: M woTSA SR REDS 4x, mt4r: M SR REDS 4x')
# Model directory
parser.add_argument('-md', '--model_path',  # 模型路径
                    type=str, default='default',
                    help='path of checkpoint for pretrained model')
# Start/End frame
parser.add_argument('-st', '--start_frame',  # 开始帧
                    type=int, default=1,
                    help='specify start frame (Start from 1)')
parser.add_argument('-ed', '--end_frame',  # 结束帧
                    type=int, default=0,
                    help='specify end frame. Default: Final frame')
# FFmpeg
parser.add_argument('-fd', '--ffmpeg_dir',  # FFmpeg路径
                    type=str, default='',
                    help='path to ffmpeg(.exe)')
parser.add_argument('-vc', '--vcodec',  # 视频编码
                    type=str, default='h264',
                    help='Video codec')
parser.add_argument('-br', '--bit_rate',  # 视频编码
                    type=str, default='100M',
                    help='Bit rate for output video')
parser.add_argument('-fps',  # 目标帧率
                    type=float,
                    help='specify fps of output video. Default: original fps * sf.')
parser.add_argument('-mc', '--mac_compatibility',  # 让苹果设备可以直接播放
                    type=bool, default=True,
                    help='If you want to play it on a mac with QuickTime or iOS, set this to True and the pixel '
                         'format will be yuv420p. ')
# Other
parser.add_argument('-bs', '--batch_size',  # Batch Size
                    type=int, default=1,
                    help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
parser.add_argument('-ec', '--empty_cache',  # Batch Size
                    type=int, default=0,
                    help='Empty cache while processing, set to 1 if you get CUDA out of memory errors; If there\'s '
                         'the process is ok, setting to 1 will slow down the process. ')
# Temporary files
parser.add_argument('-tmp', '--temp_file_path',  # 临时文件路径
                    type=str, default='tmp',
                    help='Specify temporary file path')
parser.add_argument('-rm', '--remove_temp_file',  # 是否移除临时文件
                    type=bool, default=False,
                    help='If you want to keep temporary files, select True ')

# EDVR
parser.add_argument('-net', '--net_name', type=str, default='DAIN_slowmotion',  # DAIN 的网络
                    choices=['DAIN', 'DAIN_slowmotion'], help='model architecture: DAIN | DAIN_slowmotion')

args = parser.parse_args().__dict__

model_path = {
    'EDVR': {
        'ld': 'BasicSR/experiments/pretrained_models/EDVR/EDVR_L_deblur_REDS_official-ca46bd8c.pth',
        'ldc': 'BasicSR/experiments/pretrained_models/EDVR/EDVR_L_deblurcomp_REDS_official-0e988e5c.pth',
        'l4v': 'BasicSR/experiments/pretrained_models/EDVR/EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth',
        'l4r': 'BasicSR/experiments/pretrained_models/EDVR/EDVR_L_x4_SR_REDS_official-9f5f5039.pth',
        'l4br': 'BasicSR/experiments/pretrained_models/EDVR/EDVR_L_x4_SRblur_REDS_official-983d7b8e.pth',
        'm4r': 'BasicSR/experiments/pretrained_models/EDVR/EDVR_M_woTSA_x4_SR_REDS_official-1edf645c.pth',
        'mt4r': 'BasicSR/experiments/pretrained_models/EDVR/EDVR_M_x4_SR_REDS_official-32075921.pth'
    },
    'ESRGAN': {
        'test': 'ESRGAN.ckpt'
    }
}


def listdir(folder):  # 输入文件夹路径，输出文件夹内的文件，排序并移除可能的无关文件
    disallow = ['.DS_Store', '.ipynb_checkpoints', '$RECYCLE.BIN', 'Thumbs.db', 'desktop.ini']
    files = []
    for file in os.listdir(folder):
        if file not in disallow and file[:2] != '._':
            files.append(file)
    files.sort()
    return files


class data_loader:
    def __init__(self, input_dir, input_type, start_frame):
        self.input_type = input_type
        self.input_dir = input_dir
        self.start_frame = start_frame
        self.sequence_read_funcs = {'is': cv2.imread,
                                    'npz': lambda path: numpy.load(path)['arr_0'],
                                    'npy': numpy.load
                                    }
        self.read = self.video_func if self.input_type == 'video' else self.sequence_func
        if input_type == 'video':
            self.cap = cv2.VideoCapture(input_dir)
            self.cap.set(1, self.start_frame)
            self.fps = self.cap.get(5)
            self.frame_count = int(self.cap.get(7))
            self.height = int(self.cap.get(4))
            self.width = int(self.cap.get(3))

        else:
            self.count = -1
            self.files = [f'{input_dir}/{f}' for f in listdir(input_dir)[self.start_frame:]]
            self.frame_count = len(self.files)
            self.img = self.sequence_read_funcs[input_type](self.files[0]).shape
            self.height = self.img[0]
            self.width = self.img[1]
            del self.img

        self.read = self.video_func if self.input_type == 'video' else self.sequence_func

    def video_func(self):
        return self.cap.read()

    def sequence_func(self):
        self.count += 1
        if self.count < self.frame_count:
            img = self.sequence_read_funcs[self.input_type](self.files[self.count])
            if img is not None:
                return True, img
        return False, None

    def close(self):
        if self.input_type == 'video':
            self.cap.close()


def data_writer(output_type):
    return {'tiff': lambda path, img: cv2.imwrite(path + '.tiff', img),
            'png': lambda path, img: cv2.imwrite(path + '.png', img),
            'npz': numpy.savez_compressed,
            'npy': numpy.save
            }[output_type]


def detect_input_type(input_dir):  # 检测输入类型
    if os.path.isfile(input_dir):
        if os.path.splitext(input_dir)[1].lower() == '.json':
            input_type_ = 'continue'
        else:
            input_type_ = 'video'
    else:
        files = listdir(input_dir)
        if os.path.splitext(files[0])[1].lower() == '.npz':
            input_type_ = 'npz'
        elif os.path.splitext(files[0])[1].lower() == '.npy':
            input_type_ = 'npy'
        elif os.path.splitext(files[0])[1].replace('.', '').lower() in \
                ['dpx', 'jpg', 'jpeg', 'exr', 'psd', 'png', 'tif', 'tiff']:
            input_type_ = 'is'
        else:
            input_type_ = 'mix'
    return input_type_


def check_output_dir(dire, ext=''):
    if not os.path.exists(os.path.split(dire)[0]):  # If mother directory doesn't exist
        os.makedirs(os.path.split(dire)[0])  # Create one
    if os.path.exists(dire + ext):  # If target file/folder exists
        count = 2
        while os.path.exists(f'{dire}_{count}{ext}'):
            count += 1
        dire = f'{dire}_{count}{ext}'
    else:
        dire = f'{dire}{ext}'
    if not ext:  # Output as folder
        os.mkdir(dire)
    return dire


def second2time(second: float):
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    t = '%d:%02d:%.2f' % (h, m, s)
    return t


input_type = detect_input_type(args['input'])
if input_type == 'mix':
    processes = listdir(args['input'])
    processes = [os.path.join(args['input'], process) for process in processes]
else:
    processes = [args['input']]
# Extra work
args['start_frame'] -= 1
for input_file_path in processes:
    input_type = detect_input_type(input_file_path)
    if input_type != 'continue':
        input_file_name_list = list(os.path.split(input_file_path))
        input_file_name_list.extend(os.path.splitext(input_file_name_list[1]))
        input_file_name_list.pop(1)
        temp_file_path = check_output_dir(os.path.join(args['temp_file_path'], input_file_name_list[1]))
        video = data_loader(input_file_path, input_type, args['start_frame'])
        frame_count = video.frame_count
        frame_count_len = len(str(frame_count))
        if args['fps']:
            fps = args['fps']
        elif input_type == 'video':
            fps = video.fps
        else:
            fps = 30
        # Start/End frame
        if args['end_frame'] == 0 or args['end_frame'] == frame_count or args['end_frame'] > frame_count:
            copy = True
            end_frame = frame_count
        else:
            copy = False
            end_frame = args['end_frame'] + 1
        if args['start_frame'] == 0 or args['start_frame'] >= frame_count:
            start_frame = 1
        else:
            start_frame = args['start_frame']

        if args['model_path'] == 'default':  # 模型路径
            model_path = model_path[args['algorithm']][args['mn']]
        else:
            model_path = args['model_path']

        output_type = args['output_type']
        output_dir = args['output']
        if output_dir == 'default':
            output_dir = f"{input_file_name_list[0]}/{input_file_name_list[1]}_{args['algorithm']}"
        if output_type == 'video':
            if input_file_name_list[2]:
                ext = input_file_name_list[2]
            else:
                ext = '.mp4'
        else:
            output_dir, ext = os.path.splitext(output_dir)
        output_dir = check_output_dir(output_dir, ext)
        if output_type == 'video':
            dest_path = check_output_dir(os.path.splitext(output_dir)[0], ext)
            output_dir = f'{temp_file_path}/tiff'
            output_type = 'tiff'
            os.makedirs(output_dir)
        else:
            dest_path = False

        cag = {'input_file_path': input_file_path,
               'input_type': input_type,
               'empty_cache': args['empty_cache'],
               'model_path': model_path,
               'temp_folder': temp_file_path,
               'algorithm': args['algorithm'],
               'frame_count': frame_count,
               'frame_count_len': len(str(video.frame_count)),
               'height': video.height,
               'width': video.width,
               'start_frame': start_frame,
               'end_frame': end_frame,
               'model_name': args['mn'],
               'batch_size': args['batch_size'],
               'output_type': output_type,
               'output_dir': output_dir,
               'dest_path': dest_path,
               'mac_compatibility': args['mac_compatibility'],
               'ffmpeg_dir': args['ffmpeg_dir'],
               'fps': fps,
               'vcodec': args['vcodec']
               }
        with open(f'{temp_file_path}/process_info.json', 'w') as f:
            json.dump(cag, f)
    else:
        with open(input_file_path, 'r') as f_:
            cag = json.load(f_)
        start_frame = len(listdir(cag['output_dir'])) // cag['sf']
        video = data_loader(cag['input_file_path'], cag['input_type'], start_frame - 1)

    if cag['empty_cache']:
        os.environ['CUDA_EMPTY_CACHE'] = '1'

    # Model checking
    if not os.path.exists(cag['model_path']):
        print(f"Model {cag['model_path']} doesn't exist, exiting")
        exit(1)
    # Start frame
    batch_count = (cag['frame_count'] - start_frame + 1) // cag['batch_size']
    if (cag['frame_count'] - start_frame) % cag['batch_size']:
        batch_count += 1

    # Super resolution
    SRer = SRers.__dict__[cag['algorithm']].SRer(cag['model_name'], cag['model_path'], cag['height'], cag['width'])
    SRer.init_batch(video)
    save = data_writer(cag['output_type'])
    timer = 0
    start_time = time.time()
    try:
        for i in range(batch_count):
            out = SRer.sr(video.read())
            save(f"{cag['output_dir']}/{str(i).zfill(cag['frame_count_len'])}", out)
            time_spent = time.time() - start_time
            start_time = time.time()
            if i == 0:
                initialize_time = time_spent
                print(f'Initialized and processed frame 1/{batch_count} | '
                      f'{batch_count - i - 1} frames left | '
                      f'Time spent: {round(initialize_time, 2)}s',
                      end='')
            else:
                timer += time_spent
                frames_processes = i + 1
                frames_left = batch_count - frames_processes
                print(f'\rProcessed batch {frames_processes}/{batch_count} | '
                      f"{frames_left} {'batches' if frames_left > 1 else 'batch'} left | "
                      f'Time spent: {round(time_spent, 2)}s | '
                      f'Time left: {second2time(frames_left * timer / i)} | '
                      f'Total time spend: {second2time(timer + initialize_time)}', end='', flush=True)
    except KeyboardInterrupt:
        print('\nCaught Ctrl-C, exiting. ')
        exit(256)
    del video, SRer
    print(f'\r{os.path.split(input_file_path)[1]} done! Total time spend: {second2time(timer + initialize_time)}', flush=True)
    # Video post process
    if cag['dest_path']:
        # Mac compatibility
        pix_fmt = ' -pix_fmt yuv420p' if cag['mac_compatibility'] else ''
        # Execute command
        cmd = [f"'{os.path.join(cag['ffmpeg_dir'], 'ffmpeg')}' -loglevel error ",
               f"-vsync 0 -r {cag['fps']} -pattern_type glob -i '{cag['temp_folder']}/tiff/*.tiff' ",
               f"-vcodec {cag['vcodec']}{pix_fmt} '{cag['dest_path']}'"]
        if cag['start_frame'] == 1 and cag['end_frame'] == 0:
            cmd.insert(1, '-thread_queue_size 128 ')
            cmd.insert(3, f"-vn -i '{input_file_path}' ")
        cmd = ''.join(cmd)
        print(cmd)
        os.system(cmd)
print(time.time() - everything_start_time)
