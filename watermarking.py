import cv2
import numpy as np
import pywt
from pylfsr import LFSR

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

def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    Nmax_bit = 1024
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    bits = bits.zfill(8 * ((len(bits) + 7) // 8))
    while(len(bits)<Nmax_bit):
        bits += '00100000'
    return bits

def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'

def xor1(bin_message, seq):
    # On suppose taille de bin_message est de 1024bits et seq de 4096 bits
    # Chaque bit de bin_message sera transformé en 4 bits avec un xor avec seq
    watermark = ''
    for i in range(len(bin_message)):
        it = 0
        while it < 4:
            if (bin_message[i] == '0' and seq[4*i+it]=='0') or (bin_message[i] == '1' and seq[4*i+it]=='1'):
                watermark += '0'
            else:
                watermark += '1'
            it += 1
    return watermark

def xor2(watermark, seq):
    extracted_mess = ''
    for i in range(len(watermark)):
        if (watermark[i] == '0' and seq[i]=='0') or (watermark[i] == '1' and seq[i]=='1'):
            extracted_mess += '0'
        else:
            extracted_mess += '1'
    extracted_mess = [extracted_mess[i] for i in range(0,len(extracted_mess),4)]
    extracted_mess = ''.join(extracted_mess)
    return extracted_mess

img = cv2.imread("pepper.bmp") #### Put a 0 after for dct or convert to grayscale
img_hsv = bgr2hsv(img)
img_dwt = true_dwt(img_hsv[:,:,-1])
img_dwt[0:256,0:256] = true_dwt(img_dwt[0:256,0:256]*255.0)
img_dwt[0:128,0:128] = true_dwt(img_dwt[0:128,0:128]*255.0)

"""----Generate the watermark for a message----------"""
alpha = 0.01
fc = 64
fpoly = [13,4,3,1]
message = "Le Cyrano, Versailles, France - 31/10/2019 - 20h00"

L = LFSR(fpoly=fpoly, initstate='random', verbose=False)
L.runKCycle(4096)
seq = L.seq
string_seq = [str(x) for x in seq]
string_seq = ''.join(string_seq)

bin_message = text_to_bits(message)
watermark = xor1(bin_message,string_seq)
watermark = np.array([int(x) for x in watermark]).reshape((fc,fc))

"""--------Spread Spectrum-------------"""

img_dwt_marked = np.copy(img_dwt)
img_dwt_marked[:fc,:fc] = img_dwt_marked[:fc,:fc] * (1 + alpha * watermark)

img_marked = np.copy(img_dwt_marked)
img_marked[0:128,0:128] = true_idwt(img_marked[0:128,0:128])
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
