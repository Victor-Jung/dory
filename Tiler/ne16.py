import torch
import numpy as np
import random

NE16_TP_IN = 16


def ne16_conv1x1_pad_ki(ki):
    return 16*(ki // 16 + (1 if ki % 16 != 0 else 0))


def div_and_ceil(a, b):
    return ((a - 1) // b) + 1


def ne16_weights_ki_size(ki, qw, fs1, fs2):
    return div_and_ceil(ki, NE16_TP_IN) * qw * fs1 * fs2 * 2


def ne16_weights_size(ko, ki, qw, fs1, fs2):
    return ko * ne16_weights_ki_size(ki, qw, fs1, fs2)


# assuming torch shapes, w must already be in uint format!
# format --> [Ko, KiMajor, Qw, KiMinor] (binary tensor)
#                          +++++++++++ --> these are *contiguous and packed*
def ne16_conv1x1_unroll(w, qw, format='KoKiHW', TP_IN=16):
    if format=='KoKiHW':
        pass
    elif format == 'KoHWKi':
        w = w.transpose(0, 3, 1, 2)
    else:
        raise Exception(f'Format {format} not implemented.')

    Ko, Ki, H, W = w.shape
    nb_ki = (Ki // TP_IN + (1 if Ki % TP_IN != 0 else 0))
    wbytes = np.zeros((Ko * nb_ki * qw, 2), dtype=np.uint8)
    for ko in range(Ko):
        for ki in range(Ki):
            kimaj = ki // TP_IN
            kimin = ki % TP_IN
            byte  = kimin // 8
            shift = kimin % 8
            for q in range(qw):
                index = ko*nb_ki*qw + kimaj*qw + q
                wbytes[index,byte] = np.bitwise_or(wbytes[index,byte], 1 << shift if w[ko,ki,0,0] & (1 << q) != 0 else 0)
    wbytes = wbytes.reshape(-1)
    return wbytes

def ne16_conv1x1_roll(wbytes, qw, shape, format='KoKiHW', TP_IN=16):
    if format == 'KoKiHW':
        Ko, Ki, H, W = shape
        w = np.zeros(shape, dtype=np.uint8)
        wv = w
    elif format == 'KoHWKi':
        Ko, H, W, Ki = shape
        w = np.zeros(shape, dtype=np.uint8)
        wv = w.transpose(0, 3, 1, 2)
    else:
        raise Exception(f'Format {format} not implemented.')

    nb_ki = (Ki // TP_IN + (1 if Ki % TP_IN != 0 else 0))
    for ko in range(Ko):
        for kimaj in range(nb_ki):
            for q in range(qw):
                for kimin in range(TP_IN):
                    byte  = kimin // 8
                    shift = kimin % 8
                    index = ko*nb_ki*qw*2 + kimaj*qw*2 + q*2 + byte
                    if kimaj*TP_IN+kimin < Ki:
                        wv[ko, kimaj*TP_IN+kimin, 0, 0] += (1 & (wbytes[index] >> shift)) << q
    return w

def subtile_bit_extract(subtile, bit_idx):
    retval = 0
    for i, el in enumerate(subtile):
        if el.item() & (1<<bit_idx):
            retval |= 1 << i
    return retval

def ne16_conv3x3_unroll(w, qw, format="KoKiHW", TP_IN=16, dw=False):
    if format == "KoKiHW":
        pass
    elif format == "KoHWKi":
        if dw:
            w = w.transpose(3, 0, 1, 2)
        else:
            w = w.transpose(0, 3, 1, 2)
    else:
        raise Exception(f'Format {format} not implemented.')

    Ko, Ki, H, W = w.shape
    nb_ki = (Ki // TP_IN) + (1 if Ki % TP_IN != 0 else 0)
    nb_tp_in = TP_IN // 8
    wbytes = np.zeros((Ko, nb_ki, qw, H * W, nb_tp_in), dtype=np.uint8)
    for i in range(Ko):
        for j in range(nb_ki):
            tile = w[i, j*TP_IN:(j+1)*TP_IN].transpose(1, 2, 0).reshape(H*W, -1)
            for k, subtile in enumerate(tile):
                for bit in range(qw):
                    subtile_bit = subtile_bit_extract(subtile, bit)
                    for l in range(nb_tp_in):
                        wbytes[i, j, bit, k, l] = (subtile_bit >> (l * 8)) & 0xff
    wbytes = wbytes.reshape(-1)
    return wbytes

def subtile_bit_roll(w_subtile, subtile, bit):
    s = 0
    for i, byte in enumerate(subtile):
        s += byte.item() << (i * 8)
    for i in range(w_subtile.size):
        w_subtile[i] += ((s & (1 << i)) >> i) << bit

def ne16_conv3x3_roll(wbytes, qw, shape, format="KoKiHW", TP_IN=16):
    if format == 'KoKiHW':
        Ko, Ki, H, W = shape
        w = np.zeros(shape, dtype=np.uint8)
        wv = w
    elif format == 'KoHWKi':
        Ko, H, W, Ki = shape
        w = np.zeros(shape, dtype=np.uint8)
        wv = w.transpose(0, 3, 1, 2)
    else:
        raise Exception(f'Format {format} not implemented.')

    nb_ki = (Ki // TP_IN) + (1 if Ki % TP_IN != 0 else 0)
    wbytes_reshape = wbytes.reshape(Ko, nb_ki, qw, H, W, 2)
    for i in range(Ko):
        for j in range(nb_ki):
            for bit in range(qw):
                for k in range(H):
                    for l in range(W):
                        subtile_bit_roll(wv[i, j*TP_IN:(j+1)*TP_IN, k, l].reshape(-1), wbytes_reshape[i, j, bit, k, l], bit)
    return w


if __name__ == "__main__":

    def test(name, Ko, Ki, fs, qw):
        print(f'Test {name} shape=({Ko:3}, {Ki:3}, {fs}, {fs}) qw={qw}: ', end='', flush=True)
        shape = (Ko, Ki, fs, fs)
        test_in = np.random.randint(low=0, high=1<<qw, size=shape, dtype=np.uint8)
        test_out = globals()[f'ne16_conv{fs}x{fs}_roll'](globals()[f'ne16_conv{fs}x{fs}_unroll'](test_in, qw), qw, shape)

        if not np.array_equal(test_in, test_out):
            print(f'Fail!')
            print('Test in:')
            print(test_in)
            print('Test out:')
            print(test_out)
            print(test_in[np.equal(test_in, test_out)])
        else:
            print(f'Success!')
    
    def test_generator(fs, test_count):
        print(f'Testing {fs}x{fs} convolution:')
        for i in range(test_count):
            Ko = random.randint(1, 128)
            Ki = random.randint(1, 128)
            qw = random.randint(2, 8)
            test(f'[{i}]', Ko, Ki, fs, qw)

    TEST_COUNT = 10

    test_generator(1, TEST_COUNT)
    test_generator(3, TEST_COUNT)
