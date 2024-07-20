import cv2
import os
import argparse
import glob
import numpy as np
import torch
import utils
import torch.nn as nn
from torch.autograd import Variable
from models_rddcnn import DnCNN
from skimage.io import imread, imsave
import time
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/', type=str, help='path of test dataset')
    parser.add_argument('--set_names', type=str, nargs='+', default=['BSD68'], help='test dataset names')
    parser.add_argument('--sigma', default=15, type=int, help='noise level')
    # parser.add_argument('--mode', default='S', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=0, type=int, help='save the de-noised image, 1 or 0')

    parser.add_argument('--model_path', default='models/', help='path of the model')
    parser.add_argument('--original_image', default='data/', type=str, help='path of original image')
    parser.add_argument('--noise_image', default='data/', type=str, help='path of noise image')
    parser.add_argument('--result_path', default='data/', type=str, help='result directory')
    return parser.parse_args()

def save_result(result, path):
    img = Image.fromarray(np.clip(result, 0, 1))
    print(img.mode)
    if img.mode != 'F':
      path = path if path.find('.') != -1 else path+'.tiff'
    else :
      #path = path if path.find('.') != -1 else path+'.tiff' 
      path = os.path.splitext(path)[0]+ ".tiff"  
    ext = os.path.splitext(path)[-1]
    print (path)
    
    if ext in ('.txt', '.dlm'):
      np.savetxt(path, result, fmt='%2.4f')
    else:
      img.save(path)

def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def main():
    args = parse_args()
    model = DnCNN().cuda()
    device_ids = [0]
    model = nn.DataParallel(model, device_ids=device_ids)
    if not os.path.exists(args.model_path):
        raise RuntimeError(f'model_path {0} not exist'.format(args.model_path))
    model.load_state_dict(torch.load(args.model_path))
    
    model.eval()  # evaluation mode
    if torch.cuda.is_available():
        model = model.cuda()
    with torch.no_grad():
        # for im in os.listdir(os.path.join(args.set_dir, set_cur)):
        if args.noise_image.endswith(".jpg") or args.noise_image.endswith(".bmp") or args.noise_image.endswith(".png"):

            x = np.array(imread(os.path.join(args.original_image)), dtype=np.float32)/255.0 # original_image
            np.random.seed(seed=8)  # for reproducibility
            y = np.array(imread(args.noise_image), dtype=np.float32)/255.0 # noised image
            y = y.astype(np.float32)
            start_time = time.time()
            y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])

            torch.cuda.synchronize()
            y_ = y_.cuda()
            x_ = model(y_)  # inference
            x_ = x_.view(y.shape[0], y.shape[1])
            x_ = x_.cpu()
            x_ = x_.detach().numpy().astype(np.float32)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            # print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))

            psnr_x_ = peak_signal_noise_ratio(x, x_)
            save_result(x_, path=args.result_path)  # save the denoised image
            # print(psnrs)
            print("%10s: %.2fdB " % (args.result_path, round(psnr_x_, 2)), end='')
if __name__ == "__main__":
    main()
