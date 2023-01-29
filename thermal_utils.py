from os.path import isfile
import cv2
import numpy as np


thermal_K = np.array([ 4.2943288714549999e+02, 0., 3.1111923634459998e+02, 
                     0., 4.2953142750190000e+02, 2.6612817575460002e+02, 
                     0., 0., 1. ]).reshape(3,3)
thermal_D = np.array([ -3.5808823350000002e-01, 9.9431845300000002e-02,
                     0., 0., 0. ]).reshape(1,5)

thermal_h, thermal_w = 480, 640

rgb_K = np.array([ 788.008149, 0.000000, 634.059045,
                0., 790.732322, 235.898376, 
                0., 0., 1. ]).reshape(3,3)
rgb_D = np.array([ -0.050724, 0.036578,
       -0.012903, 0.010071, -6.3505208e-02 ]).reshape(1,5)

rgb_h, rgb_w = 560, 1280


def intensity2temp(intensity):
    PB = 1428
    PF = 1.0
    PO = 118.126
    PR = 377312.0
    temp = PB/np.log(PR/(intensity - PO) + PF) - 273.15
    return temp


def temp2intensity(temperature):
    PB = 1428
    PF = 1.0
    PO = 118.126
    PR = 377312.0
    intensity = PR/(np.exp(PB/(273.15 + temperature)) - PF) + PO
    return intensity


def _undistortImage(cv_img, K, D):
    return cv2.undistort(cv_img, K, D, None)


def _colorizeImage(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
    return rgb_img


def _cropImage(img, h, w):    
    if img.ndim == 3:
        return img[0:h, 0:w, :]
    elif img.ndim == 2:
        return img[0:h, 0:w]
    else:
        print('Unknown image channel dimensions: {}.'.format(img.ndim))
        raise RuntimeError


def _normalizeImageWithMeanStd(img, sigma=1.0, keeptype=False):
    mean, std = np.mean(img), np.std(img)
    min, max = mean - sigma* std, mean + sigma* std
    
    img = (img - min) / (max - min)
    np.clip(img, 0, 1, out=img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img if keeptype else img.astype('uint8')


def _normalizeImageWithMinMax(img, keeptype=False):
    min, max = img.min(), img.max()
    img = (img - min) / (max - min)
    np.clip(img, 0, 1, out=img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img if keeptype else img.astype('uint8')


def _normalizeImageHard(img, keeptype=False):
    return img / 64 if keeptype else (img / 64).astype('uint8')


def _normalizeImageWithExternalMinMax(img, min_T, max_T, keeptype=False):
    min = temp2intensity(min_T)
    max = temp2intensity(max_T)
    img = (img - min) / (max - min)
    np.clip(img, 0, 1, out=img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img if keeptype else img.astype('uint8')


def _binarizeImageWithExternalMinMax(img, min_T, max_T, keeptype=False):
    min = temp2intensity(min_T)
    max = temp2intensity(max_T)
    img = np.where((img >= min) & (img <= max), 255, 0)
    return img if keeptype else img.astype('uint8')


def _enhanceImageWithHistogram(img, bin=30, clahe=True):
    min_, max_ = np.min(img), np.max(img)
    h,w = img.shape[:2] if img.ndim == 3 else img.shape
    total = h*w
    h_i = np.append(np.arange(min_, max_, (max_-min_)/bin), 2**14)
    n_i = np.array([np.count_nonzero((img >= h_i[i]) & (img <= h_i[i+1])) for i in range(bin)])

    img_tmp = np.zeros((bin,h,w))
    for i in range(bin) : 
        pixel_interest = (img >= h_i[i]) & (img < h_i[i+1])
        if n_i[i] > 0:
            img_tmp[i,...] = pixel_interest * (img - h_i[i] + np.sum(n_i[0:i])) / total
    img_enhanced = img_tmp.sum(axis=0)
    img_enhanced = (img_enhanced*255).astype('uint8')
    
    if clahe:
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
        img_enhanced = clahe.apply(img_enhanced)
    return img_enhanced


def _smoothImage(img, kernel_size=3):
    return cv2.GaussianBlur(img, (kernel_size,kernel_size), kernel_size)


def _transformPixel(img, alpha=1.0, beta=0.0):
    if np.max(img) <= 255:
        print('processing 8-bit image')
        return np.clip(alpha * img + beta, 0, 255)
    elif np.max(img) <= 2**14-1:
        print('processing 14-bit image')
        return np.clip(alpha * img + beta, 0, 2**14-1)


def _gammaCorrection(img, gamma=1.0):
    if img.max() <= 255:
        print('processing 8-bit image')
        return np.clip(np.power(img/255.0, gamma)*255.0, 0, 255)
    elif img.max() <= 2**14-1:
        print('processing 14-bit image')
        return np.clip(np.power(img/(2**14-1), gamma)*(2**14-1), 0, 2**14-1)


def processThermalImage(img, undistort=False, K=thermal_K, D=thermal_D, 
                crop=False, h=thermal_h, w=thermal_w, 
                colorize=False, smooth=False, kernel_size=3, save_path=None, 
                normalize_meanstd=False, sigma=1.0,
                normalize_minmax=False, normalize_hard=False,
                normalize_external=False, min_T=0.0, max_T=30.0,
                binarize_minmax=False, keeptype=False,
                enhance_histogram=False, bin_size=30, clahe=True,
                pixel_transform=False, alpha=1.0, beta=0.0, gamma=1.0):

    if isinstance(img, str) and isfile(img):
        img = cv2.imread(img, -1)

    if [normalize_meanstd, normalize_minmax, normalize_external, normalize_hard].count(True) > 1:
        print("Only one normalization method should be set.")
        raise RuntimeError
    if min_T > max_T or max_T > 100.0 or min_T <= -273.15:
        print("Invalid min-max temperature, set min-max below 100")
    if undistort is True and (K is None or D is None):
        print("K and D should be set in order to undistort image.")
        raise RuntimeError
    if crop is True and (h is None or w is None):
        print("width and height should be set in order to crop image.")
        raise RuntimeError

    if colorize:
        img = _colorizeImage(img)
    if undistort:
        img = _undistortImage(img, K, D)
    if crop:
        img = _cropImage(img, h, w)
    if smooth:
        img = _smoothImage(img, kernel_size)
    if binarize_minmax:
        img = _binarizeImageWithExternalMinMax(img, min_T, max_T, keeptype)
    if normalize_meanstd:
        img = _normalizeImageWithMeanStd(img, sigma, keeptype)
    elif normalize_minmax:
        img = _normalizeImageWithMinMax(img, keeptype)
    elif normalize_external:
        img = _normalizeImageWithExternalMinMax(img, min_T, max_T, keeptype)
    elif normalize_hard:
        img = _normalizeImageHard(img, keeptype)
    if enhance_histogram:
        img = _enhanceImageWithHistogram(img, bin_size, clahe)
    if pixel_transform:
        img = _gammaCorrection(img, gamma)
        img = _transformPixel(img, alpha, beta)

    if save_path is not None and cv2.imwrite(save_path, img) is False:
        print("Cannot save image on path {}".format(save_path)) 

    return img


def processRGBImage(img, undistort=False, K=rgb_K, D=rgb_D, 
                crop=False, h=rgb_h, w=rgb_w, 
                colorize=False, smooth=False, kernel_size=3, save_path=None):

    if isinstance(img, str) and isfile(img):
        img = cv2.imread(img, -1)

    if undistort is True and (K is None or D is None):
        print("K and D should be set in order to undistort image.")
        raise RuntimeError
    if crop is True and (h is None or w is None):
        print("width and height should be set in order to crop image.")
        raise RuntimeError

    if colorize:
        img = _colorizeImage(img)
    if undistort:
        img = _undistortImage(img, K, D)
    if crop:
        img = _cropImage(img, h, w)
    if smooth:
        img = _smoothImage(img, kernel_size)

    if save_path is not None and cv2.imwrite(save_path, img) is False:
        print("Cannot save image on path {}".format(save_path)) 

    return img