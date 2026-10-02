import numpy as np
from scipy import fft

def calc_G(source_coords, receiver_coords, c, dt, n_fft):
    src = np.asarray(source_coords, np.float32)
    rec = np.asarray(receiver_coords, np.float32)
    d = np.linalg.norm(rec[:, None] - src[None], axis=-1)
    f = fft.rfftfreq(n_fft, dt).astype(np.float32)
    kr = (np.float32(2 * np.pi) / np.float32(c)) * f[:, None, None] * d
    return np.exp(kr * np.complex64(-1j)) / d

def simulate(signals, source_coords, receiver_coords, c, dt, backward=False):
    X = np.asarray(signals, np.float32)
    n = len(X)
    n_fft = 2 * n
    Xf = fft.rfft(X, n=n_fft, axis=0)
    G = calc_G(source_coords, receiver_coords, c, dt, n_fft)
    if backward:
        Yf = np.einsum("frs,fr->fs", G.conj(), Xf)
    else:
        Yf = np.einsum("frs,fs->fr", G, Xf)
    return fft.irfft(Yf, n=n_fft, axis=0)[:n]
