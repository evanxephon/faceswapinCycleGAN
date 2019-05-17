# code borrowed from https://github.com/shaoanlu/faceswap-GAN/blob/master/data_loader/data_augmentation.py
import numpy as np
import cv2
from umeyama import umeyama
from scipy import ndimage
from pathlib import PurePath, Path

ROTATION_RANGE = 10
ZOOM_RANGE = 0.1
SHIFT_RANGE = 0.05
RANDOM_FLIP = 0.5

def get_motion_blur_kernal(size=7):
    
    angle = np.random.uniform(-180,180)
    
    kernel = np.zeros((size,size))
    kernel[int((size-1)//2), :] = np.ones(size)
    kernel = ndimage.interpolation.rotate(kernel, angle, reshape=False)
    kernel = np.clip(kernel, 0, 1)
    
    nomalize_factor = 1 / np.sum(kernel)
    kernel = kernel * noramlize_factor
    
    return kernel

def motion_blur(images):
    
    blur_size = np.random.choice([5,7,9,11])
    
    kernel_motion_blur = get_motion_blur_kernal(blur_size)
    
    for i in range(len(image)):
        images[i] = cv2.filter2D(images[i], -1, kernel_motion_blur).astype(np.float64)
    
    return images

def random_transform(image, rotation_range=ROTATION_RANGE, zoom_range=ZOOM_RANGE, shift_range=SHIFT_RANGE, random_flip=RANDOM_FLIP):
    
    h,w = image.shape[:2]
    
    #rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    
    xshift = np.random.uniform(-shift_range, shift_range) * w
    yshift = np.random.uniform(-shift_range, shift_range) * h
    
    mat = cv2.getRotationMatrix2D((w//2, h//2), rotation, scale)
    mat[:, 2] += (xshift, yshift)
    
    result = cv2.warpAffine(image, mat, (w,h), borderMode=cv2.BORDER_REPLICATE)
    
    #if np.random.random() < random_flip:
    #    result = result[:,::-1,:]
        
    return result

def random_warp_rev(image, res=64, roi=0.6):
    
    assert image.shape == (256,256,3)
    resize_scale = res//64
    
    assert resize_scale >= 1
    
    interp_param = 80 * resize_scale
    interp_slice = slice(interp_param//10, 9*interp_param//10)
    
    dst_pnts_slice = slice(0, 65*resize_scale, 16*resize_scale)
    
    rand_coverage = np.random.randint(128 * roi / 4) + 128 * roi
    rand_scale = np.random.uniform(5., 6.2)
    
    warp_range = np.linspace(128 - rand_coverage, 128 + rand_coverage, 5)
    
    mapx = np.broadcast_to(warp_range, (5,5))
    mapy = maxp.T
    
    mapx = mapx + np.random.normal(size=(5,5), scale=rand_scale)
    mapy = mapy + np.random.normal(size=(5,5), scale=rand_scale)
    
    interp_mapx = cv2.resize(mapx, (interp_param, interp_param))[interp_slice, interp_slice].astype('float32')
    interp_maxy = cv2.resize(mapy, (interp_param, interp_param))[interp_slice, interp_slice].astype('float32')
    
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    
    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[dst_pnts_slice, dst_pnts_slice].T.reshape(-1,2)
    
    mat = umeyama(src_points, dst_points, True)[:2]
    
    real_image = cv2.warpAffine(image, mat, (res, res))
    
    return warped_image, real_image

def warp_and_aug(image, config):
    
    image = random_transform(image)
    
    warped_image, real_img = random_warp_rev(image, roi=0.8)
    
    warped_image, real_img = motion_blur([warped_image, real_img])
    
    return warped_img, real_img
