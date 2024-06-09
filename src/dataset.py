import torch
import numpy as np
import tensorflow as tf
from tqdm import tqdm

class MultiChannelDataset(torch.utils.data.Dataset):

    def __init__(self, soi_gen_fn, sig_inf, sig_len, Nchannels, N, sinr_range, soi_aoa=None, freqoffset_std=0, return_bits=False, fix=False):
        self.soi_gen_fn = soi_gen_fn
        self.sig_inf = sig_inf
        self.sig_len = sig_len
        self.Nchannels = Nchannels
        self.Array = ArrayReceiver(Nchannels)
        self.N = N
        self.sinr = sinr_range
        self.soi_aoa = soi_aoa # az, el
        self.fo_std = freqoffset_std
        self.return_bits = return_bits
        self.fix = False
        if fix:
            self.data = []
            for ii in tqdm(range(N), desc='Preparing fixed data'):
                self.data.append(self.__getitem__(ii))
        self.fix = fix

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        if self.fix: return self.data[index]
        with tf.device('CPU'):
            sig_target, _, bits, _ = self.soi_gen_fn(1, self.sig_len)
            sig_target = sig_target[0, :self.sig_len].numpy()
        sig_interference = self.sig_inf[np.random.randint(self.sig_inf.shape[0]), :]
        start_idx = np.random.randint(sig_interference.shape[0]-self.sig_len)
        sig_interference = sig_interference[start_idx:start_idx+self.sig_len]

        # carrier frequency offset
        if self.fo_std > 0:
            cfo = np.random.randn()*self.fo_std
            sig_interference *= np.exp(1j*2*np.pi*np.arange(self.sig_len)*cfo)

        # Interference Coefficient
        sinr = self.sinr[0] + (self.sinr[1] - self.sinr[0])*np.random.rand()
        coeff = np.sqrt(10**(-sinr/10)) * np.exp(1j*2*np.pi*np.random.rand())

        ang = self.Array.get_random_angles(2)
        if self.soi_aoa is not None: # fix AoA of SOI
            ang[0,:] = np.array(self.soi_aoa)
        sig_mixture = self.mix(sig_target, sig_interference * coeff, ang)
        if self.return_bits:
            return torch.from_numpy(sig_mixture), torch.from_numpy(sig_target), np.squeeze(bits.numpy()), sinr, ang
        else:
            return torch.from_numpy(sig_mixture), torch.from_numpy(sig_target)
        
    def mix(self, x, y, ang):
        steering_vec = self.Array.get_steering_vectors(ang)
        #steering_vec /= steering_vec[0,0] # correct phase for SOI in first component
        return steering_vec @ np.stack([x, y], axis=0)
        

class ArrayReceiver:
    '''
    This class implements antenna array reception of multiple narrowband
    signals with specified angles of arrival (linear mixture Y = AX).

    Parameters
    ----------
    N : int
        Number of antennas
    '''

    def __init__(self, N):
        # normalized antenna positions with lambda/2 spacing
        # x, y, z components
        if N == 1:
            self.ant_pos = np.array([[0, 0, 0]], dtype=np.float32)
        elif N == 2: 
            self.ant_pos = np.array(
                [[   0, 0, 0], 
                 [ 0.5, 0, 0]], dtype=np.float32
            )
        elif N == 4:
            self.ant_pos = np.array(
                [[  0,   0, 0],
                 [0.5,   0, 0],
                 [0.5, 0.5, 0],
                 [  0, 0.5, 0]], dtype=np.float32
            )
    
    def get_steering_vectors(self, angles):
        '''
        Compute steering vectors for given angles of incidence.

        Arguments
        ---------
        angles : 2d array
            numpy array of shape (nsig, 2) containing angles of arrival in the order (az, el)

        Returns
        -------
        steering_vec: 2d array
            steering vectors of shape (nant, nsig)
        '''
        nsig = angles.shape[0]
        az = angles[:,0]
        el = angles[:,1]
        
        # wave vectors
        wave_vec = np.zeros((3, nsig), dtype=np.float32)
        wave_vec[0,:] = np.sin(az) * np.cos(el) #np.cos(az)*np.cos(el)
        wave_vec[1,:] = np.cos(az) * np.cos(el) #np.sin(az)*np.cos(el)
        wave_vec[2,:] = np.sin(el)
        wave_vec = wave_vec
        
        # steering vectors
        steering_vec = np.exp(2.0*np.pi*1j * self.ant_pos @ wave_vec)
        return steering_vec.astype(np.complex64)
        
    @staticmethod
    def get_random_angles(num_angles):
        '''
        Generate random angles of arrival (azimuth, elevation) of shape (num_angles, 2).
        '''
        az = np.random.rand(num_angles) * 2.0*np.pi
        el = 15/360*np.random.randn(num_angles) * 2.0*np.pi #TODO?
        return np.stack([az, el], axis=-1)

if __name__ == '__main__':
    ar = ArrayReceiver(4)
    sv = ar.get_steering_vectors(np.array([[0,0],[np.pi/4,0],[np.pi/2,0]]))
    print(sv)