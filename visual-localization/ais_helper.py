import torch
import numpy as np
import scipy
import random
import cv2
import re
from random import randint
import yaml
import sys


class CameraParams:
    def __init__(self, cal_file, full_parse=True):
        self.c_m_0 = np.zeros((3,3))
        self.c_m_1 = np.zeros((3, 3))
        self.dist_0 = np.zeros((4,1))
        self.dist_1 = np.zeros((4,1))
        self.R = np.zeros((3,3))
        self.T = np.zeros((3,1))
        self.R_1 = np.zeros((3,3))
        self.R_2 = np.zeros((3, 3))
        self.P_1 = np.zeros((3, 4))
        self.P_2 = np.zeros((3, 4))
        self.baseline = 0.0

        stream = open(cal_file, 'r')
        self.c_file = yaml.load(stream)
        self.full_parse = full_parse
        self.prepare()

    def prepare(self):

        lp = self.c_file["left"]
        rp = self.c_file["right"]

        self.c_m_0 = np.matrix(
            [[lp["intrinsics"][0], 0, lp["intrinsics"][2]], [0, lp["intrinsics"][1], lp["intrinsics"][3]], [0, 0, 1]])

        self.c_m_1 = np.matrix(
            [[rp["intrinsics"][0], 0, rp["intrinsics"][2]], [0, rp["intrinsics"][1], rp["intrinsics"][3]], [0, 0, 1]])

        if 'baseline' in self.c_file:
            self.baseline = self.c_file['baseline']

        if self.full_parse:
            self.dist_0 = np.array([lp["distortion_coeffs"][0], lp["distortion_coeffs"][1], lp["distortion_coeffs"][2],
                                    lp["distortion_coeffs"][3]])

            self.dist_1 = np.array([rp["distortion_coeffs"][0], rp["distortion_coeffs"][1], rp["distortion_coeffs"][2],
                                    rp["distortion_coeffs"][3]])

            self.R = np.matrix(rp["T_cn_cnm1"])[0:3, 0:3]
            self.T = np.array(np.matrix(rp["T_cn_cnm1"])[0:3, 3])
            self.T_bl_cam = np.matrix(rp["T_cn_cnm1"])
            self.resolution = tuple(lp["resolution"])

class ColorCode():

    def random_color_coding(self, max_label):
        coding = {}
        for i in range(max_label + 1):
            coding[i] = [randint(0, 255), randint(0, 255), randint(0, 255)]
        return coding

    def color_code_labels(self, net_out, argmax=True):
        if argmax:
            labels, indices = net_out.max(1)
            labels_cv = indices.cpu().data.numpy().squeeze()
        else:
            labels_cv = net_out.cpu().data.numpy().squeeze()

        h = labels_cv.shape[0]
        w = labels_cv.shape[1]

        color_coded = np.zeros((h, w, 3), dtype=np.uint8)

        for x in range(w):
            for y in range(h):
                color_coded[y, x, :] = self.color_coding[labels_cv[y, x]]

        return color_coded

    def __init__(self, max_classes):
        super(ColorCode, self).__init__()
        self.color_coding = self.random_color_coding(max_classes)

def buildColorMap():
    colors = []
    colors.append((0, np.array([255,255,255])))
    colors.append((7, np.array([255,255,255])))
    colors.append((1, np.array([255, 0, 0])))
    colors.append((2, np.array([0, 255, 255])))
    colors.append((3, np.array([120, 60, 100])))
    colors.append((4, np.array([200, 180, 120])))
    colors.append((5, np.array([80, 100, 200])))
    colors.append((6, np.array([100, 255, 50])))
    colors.append((8, np.array([200, 100, 150])))
    colors.append((9, np.array([255, 150, 0])))
    colors.append((10, np.array([200, 80, 200])))

    color_dict = dict(colors)
    return color_dict


# functions to show an image
def decodeImage(img, channels=3, color_map=None):
    npimg = img

    if (channels > 1):
        npimg = np.squeeze(npimg)
        npimg = np.transpose(npimg, (1, 2, 0))

    if (channels == 1):
        if (npimg.ndim > 3):
            npimg = np.squeeze(npimg, 0)
    if (color_map is not None):
        npimg = colorizeLabel(npimg, color_map)
        npimg = np.transpose(npimg, (1, 2, 0))
    npimg = npimg.squeeze()

    return npimg

def colorizeLabel(net_out, color_dict = None, argmax = True):
    if argmax:
        labels, indices = net_out.max(1)
        labels_cv = indices.cpu().data.numpy().squeeze()
    else:
        labels_cv = net_out.cpu().data.numpy().squeeze()

    h = labels_cv.shape[0]
    w = labels_cv.shape[1]

    color_coded = np.zeros((h, w, 3), dtype=np.uint8)

    for x in range(w):
        for y in range(h):
            if labels_cv[y, x] in color_dict:
                color_coded[y, x, :] = color_dict[labels_cv[y, x]]

    return color_coded


def labelFromTensor(in_tensor, color_map):
    if (in_tensor.is_cuda):
        var = in_tensor.cpu()
    else:
        var = in_tensor

    tensor = var.data.numpy()
    tensor = np.argmax(tensor, 1)
    return decodeImage(tensor, 1, color_map)

#def loadNumpyBlob(path, width, height):
#   print ("Loading mean %s" % path)
#   blob = caffe.proto.caffe_pb2.BlobProto()
#   data = open( path, 'rb' ).read()
#   blob.ParseFromString(data)
#   arr = np.array( caffe.io.blobproto_to_array(blob) )
#   mean_np = scipy.misc.imresize(arr[0],(height,width))
#   mean_np = np.rollaxis(mean_np,2)
#   return torch.from_numpy(mean_np).float()

def cropRois(tensor_np, feature_map, min_area=100):
    crops = []
    for i in range(tensor_np.shape[0]):
        s = tensor_np[i]
        x = s[0]
        y = s[1]
        w = s[2]
        h = s[3]
        if (w * h < min_area):
            continue
        crop = feature_map[:, :, y:y + h, x:x + w]
        crops.append(crop)

        # crop = crop.cpu().data.numpy().astype("uint8").squeeze()
        # crop = np.rollaxis(crop,0,3)
        # cv2.imshow("in", crop.squeeze())
        # cv2.waitKey()

    return crops


def getSeedLabel(seed, label):
    max_instance = np.amax(label)
    for i in range(1, max_instance):
        c_label = np.count_nonzero((label == [i]).all(axis=0))


def visFlow(in_flow, name, show_magnitude = True):
    # print np.max(in_flow)
    x = in_flow[..., 0]
    y = in_flow[..., 1]
    mag, ang = cv2.cartToPolar(x, y)
    mag_disp = mag
    cv2.normalize(mag_disp, mag_disp, 0, 255, cv2.NORM_MINMAX)
    mag_disp = mag_disp.astype('uint8')
    mag_disp = cv2.applyColorMap(mag_disp, cv2.COLORMAP_JET)

    hsv = np.zeros((in_flow.shape[0], in_flow.shape[1], 3)).astype('uint8')
    hsv[..., 2] = 255
    hsv[..., 0] = (ang * (180/np.pi))/2.0
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if show_magnitude:
        cv2.imshow(name + "mag_disp", mag_disp)
    cv2.imshow(name , bgr)
    return bgr


def visDepth(depth, name):
    depth_clipped = np.zeros(depth.shape, dtype=depth.dtype)
    np.clip(depth, 0, 30, depth_clipped)
    depth_norm = np.zeros(depth_clipped.shape)
    depth_norm = cv2.normalize(depth_clipped, depth_norm, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    cv2.imshow(name, depth_color)
    return depth_color


def load_flo(file):
    magic = np.fromfile(file, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(file, np.int32, count=1)[0]
        h = np.fromfile(file, np.int32, count=1)[0]
        print("Reading %d x %d flo file" % (h, w))
        data2d = np.fromfile(file, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
        file.close()
    return data2d


def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.reshape(data, shape), scale


'''
Save a Numpy array to a PFM file.
'''


def save_pfm(file, image, scale=1):
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)
