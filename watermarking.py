import cv2
import numpy as np
import pywt

def bgr2hsv(img):
    img_hsv = np.copy(img)
    img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2HSV)
    return img_hsv

def hsv2bgr(img):
    img_bgr = np.copy(img)
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_HSV2BGR)
    return img_bgr

def dct(img):
    img_dct = np.copy(img)
    img_dct = np.float32(img_dct)/255.0
    img_dct = cv2.dct(img_dct)
    return img_dct

def idct(img_dct):
    return cv2.idct(img_dct)

def dwt(img):
    lena_dct = dct(img)

    lena_idct = np.copy(lena_dct)
    lena_idct[0:256,0:256] = cv2.idct(lena_idct[0:256,0:256])
    lena_idct[0:256,256:512] = cv2.idct(lena_idct[0:256,256:512])
    lena_idct[256:512,0:256] = cv2.idct(lena_idct[256:512,0:256])
    lena_idct[256:512,256:512] = cv2.idct(lena_idct[256:512,256:512])

    lena_idct[0:256,0:256] = cv2.dct(lena_idct[0:256,0:256])
    lena_idct[0:128,0:128] = cv2.idct(lena_idct[0:128,0:128])
    lena_idct[0:128,128:256] = cv2.idct(lena_idct[0:128,128:256])
    lena_idct[128:256,0:128] = cv2.idct(lena_idct[128:256,0:128])
    lena_idct[128:256,128:256] = cv2.idct(lena_idct[128:256,128:256])
    return lena_idct

def idwt(img):
    result = np.copy(img)
    result[0:128,0:128] = dct(img[0:128,0:128])
    result[0:128,128:256] = dct(img[0:128,128:256])
    result[128:256,0:128] = dct(img[128:256,0:128])
    result[128:256,128:256] = dct(img[128:256,128:256])

    result[0:256,256:512] = dct(img[0:256,256:512])
    result[256:512,0:256] = dct(img[256:512,0:256])
    result[256:512,256:512] = dct(img[256:512,256:512])
    return result

def true_dwt(img):
    img_dwt = np.copy(img)
    img_dwt = np.float32(img_dwt)/255.0
    LL, (LH, HL, HH) = pywt.dwt2(img_dwt,'haar')
    img_dwt[0:img.shape[0]//2, 0:img.shape[0]//2] = LL
    img_dwt[0:img.shape[0]//2, img.shape[0]//2:img.shape[0]] = LH
    img_dwt[img.shape[0]//2:img.shape[0], 0:img.shape[0]//2] = HL
    img_dwt[img.shape[0]//2:img.shape[0], img.shape[0]//2:img.shape[0]] = HH
    return img_dwt

def true_idwt(img):
    (LL, (LH, HL, HH)) = (img[0:img.shape[0]//2, 0:img.shape[0]//2], (img[0:img.shape[0]//2, img.shape[0]//2:img.shape[0]], img[img.shape[0]//2:img.shape[0], 0:img.shape[0]//2], img[img.shape[0]//2:img.shape[0], img.shape[0]//2:img.shape[0]]))
    return pywt.idwt2((LL, (LH, HL, HH)), 'haar')

img = cv2.imread("pepper.bmp") #### Put a 0 after for dct or convert to grayscale
img_hsv = bgr2hsv(img)
img_dwt = true_dwt(img_hsv[:,:,-1])
img_dwt[0:256,0:256] = true_dwt(img_dwt[0:256,0:256]*255.0)
#img_dwt[0:128,0:128] = true_dwt(img_dwt[0:128,0:128]*255.0)

"""--------Spread Spectrum-------------"""
alpha = 0.01
fc = 128
watermark = np.random.rand(fc,fc)
watermark[0,0] = 0

img_dwt_marked = np.copy(img_dwt)
img_dwt_marked[:fc,:fc] = img_dwt_marked[:fc,:fc] * (1 + alpha * watermark)

img_marked = np.copy(img_dwt_marked)
#img_marked[0:128,0:128] = true_idwt(img_marked[0:128,0:128])
img_marked[0:256,0:256] = true_idwt(img_marked[0:256,0:256])
img_marked = true_idwt(img_marked)

img_hsv_marked = np.copy(img_hsv)
img_hsv_marked[:,:,-1] = img_marked*255
img_bgr_marked = hsv2bgr(img_hsv_marked)
cv2.imshow("Img Original", img)
#cv2.imshow("Img HSV", img_hsv)
#cv2.imshow("Img BGR from HSV", img_bgr)
#cv2.imshow("Img DWT", img_dwt)
cv2.imshow("The watermark", watermark)
cv2.imshow("Img Watermarked DWT", img_dwt_marked)
cv2.imshow("Img Watermarked", img_marked)
#cv2.imshow("Img HSV + Watermarked", img_hsv_marked)
cv2.imshow("Img BGR + Watermarked", img_bgr_marked)
cv2.waitKey(0)

######## Reading a video ###########
# cap = cv2.VideoCapture("Nom-video.mp4")
# while (cap.isOpened()):
    # _,frame = cap.read()

    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break

# cap.release()
# cv2.destroyAllWindows()
