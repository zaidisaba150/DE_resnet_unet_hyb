import argparse
import cv2
import torch
from network import ResnetUnetHybrid
import image_utils
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


def predict_img(img_folder, output_folder):
    """Inference a single image."""
    # switch to CUDA device if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    # load model
    print('Loading model...')
    model = ResnetUnetHybrid.load_pretrained(device=device)
    model.eval()

    i = 0
    for images in os.listdir(img_folder):

      img_path = os.path.join(img_folder,images)
      output_path = os.path.join(output_folder,os.path.splitext(images)[0]+'.npy')
      # load image
      img = cv2.imread(img_path)[..., ::-1]
      img = image_utils.scale_image(img)
      img = image_utils.center_crop(img)
      inp = image_utils.img_transform(img)
      inp = inp[None, :, :, :].to(device)

      i+=1
      # inference
      print(str(i)+'/'+str(len(os.listdir(img_folder)))+" images converted")
      output = model(inp)

      # transform and plot the results
      output = output.cpu()[0].data.numpy()
      # plt.figure()
      pred = np.transpose(output, (1, 2, 0))
      # plt.imshow(pred[:, :, 0])
      np.save(output_path,pred)


def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_folder', required=True, type=str, help='Path to the input folder.')
    parser.add_argument('-o', '--output_folder', required=True, type=str, help='Path to the output folder.')
    return parser.parse_args()


def main():
    args = get_arguments()
    predict_img(args.img_folder, args.output_folder)


if __name__ == '__main__':
    main()
