from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import cv2

plt.rcParams['figure.figsize'] = [16,16]
plt.rcParams.update({'font.size':18})

img = imread('result2/hist/1.jpg')

#Wavelet Compression
n = 4
w = 'haar'
coeffs = pywt.wavedec2(img, wavelet = w, level = n)

coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

Csort = np.sort(np.abs(coeff_arr.reshape(-1)))

for keep in(0.1, 0.05, 0.01, 0.005):
    threshold = Csort[int(np.floor((1-keep)*len(Csort)))]
    ind = np.abs(coeff_arr) > threshold
    Cfilt = coeff_arr * ind

    coeffs_filt = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format='wavedec2')

    #plot reconstruction
    recon = pywt.waverec2(coeffs_filt, wavelet = w)
    plt.figure()
    plt.imshow(recon.astype('uint8'), cmap='gray')
    plt.axis('off')
    plt.rcParams['figure.figsize'] = [16,16]
    plt.title('keep = ' +str(keep))

    path = os.path.join('result2/dwtcom/keep' + str(keep*100) +'%')
    plt.imsave(path + '.jpg', recon.astype('uint8'), cmap = 'gray')
