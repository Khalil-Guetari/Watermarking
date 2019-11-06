import numpy as np
from pylfsr import LFSR

"""
def str2bin(message):
    #bin = ''.join(format(ord(x), 'b') for x in message)
    bin = bin(int(binascii.hexlify('message'), 16))
    return bin

def decode_binary_string(s, encoding='UTF-8'):
    byte_string = ''.join(chr(int(s[i*8:i*8+8],2)) for i in range(len(s)//8))
    return byte_string
"""

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

fpoly = [13,4,3,1]

L = LFSR(fpoly=fpoly, initstate='random', verbose=False)
L.runKCycle(4096)
seq = L.seq
string_seq = [str(x) for x in seq]
string_seq = ''.join(string_seq)

#seq = seq.reshape((64,64))
#seq = np.array(seq)

mess = "Le Cyrano, Versailles, France - 31/10/2019 - 20h00"

bin = text_to_bits(mess)

watermark = xor1(bin,string_seq)
watermark_array = np.array([int(x) for x in watermark]).reshape((64,64))
print(watermark_array)
#extracted_mess = xor2(watermark, string_seq)

#final_mess = text_from_bits(extracted_mess)

#print(final_mess)
