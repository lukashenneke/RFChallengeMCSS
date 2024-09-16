import torch
import numpy as np
import tensorflow as tf
from tqdm import tqdm

intf_to_noise_ratio_dB = 15

class MultiChannelDataset(torch.utils.data.Dataset):

    def __init__(self, soi_gen_fn, sig_inf, sig_len, Nchannels, Ndata, sinr_range, soi_aoa=None, freqoffset_std=0, return_bits=False, fix=False, array_type='ULA'):
        self.soi_gen_fn = soi_gen_fn
        self.sig_inf = sig_inf
        self.sig_len = sig_len
        self.Nchannels = Nchannels
        self.array_type = array_type
        self.Array = ArrayReceiver(Nchannels, array_type)
        self.Ndata = Ndata
        self.sinr = sinr_range
        self.soi_aoa = soi_aoa # az, el
        self.fo_std = freqoffset_std
        self.return_bits = return_bits
        self.fix = False
        if fix:
            self.data = []
            for ii in tqdm(range(Ndata), desc='Preparing fixed data'):
                self.data.append(self.__getitem__(ii))
        self.fix = fix

    def __len__(self):
        return self.Ndata

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

        # interference coefficient
        sinr = self.sinr[0] + (self.sinr[1] - self.sinr[0])*np.random.rand()
        coeff = np.sqrt(10**(-sinr/10)) * np.exp(1j*2*np.pi*np.random.rand())

        ang = self.Array.get_random_angles(2)
        if self.soi_aoa is not None: # fix AoA of SOI
            ang[0,:] = np.array(self.soi_aoa)
        sig_mixture = self.mix(sig_target, sig_interference * coeff, ang)

        # add noise
        noise = np.random.randn(self.Nchannels, self.sig_len, 2).astype(np.float32)/np.sqrt(2)
        noise = noise[...,0] + 1j*noise[...,1]
        noise *= np.sqrt(np.mean(np.abs(sig_interference*coeff)**2) / 10**(intf_to_noise_ratio_dB/10))
        sig_mixture += noise

        if self.return_bits:
            return torch.from_numpy(sig_mixture), torch.from_numpy(sig_target), np.squeeze(bits.numpy()), sinr, ang
        else:
            return torch.from_numpy(sig_mixture), torch.from_numpy(sig_target)
        
    def mix(self, x, y, ang):
        steering_vec = self.Array.get_steering_vectors(ang)
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

    def __init__(self, N: int, array: str='ULA', d: float=0.5):
        # x, y, z components
        # d: normalized spacing of neighbouring antennas ~ lambda/2 
        self.ant_pos = np.zeros((N,3), dtype=np.float32)
        if array.upper() == 'ULA':
            self.ant_pos[:,0] = np.arange(N) * d 
        elif array.upper() == 'UCA':
            r = d / (np.sqrt(2.0) * np.sqrt(1.0 - np.cos(2*np.pi/N))) # TODO
            arr = np.arange(N)/N * 2*np.pi
            self.ant_pos[:,0] = r - r * np.cos(arr)
            self.ant_pos[:,1] = - r * np.sin(arr)
        elif array.upper() == 'URA' and (float(np.sqrt(N))).is_integer():
            Nsq = int(np.sqrt(N))
            self.ant_pos[:,0] = (np.arange(N) % Nsq) * d
            self.ant_pos[:,1] = np.repeat(np.arange(Nsq), Nsq) * d
        else:
            raise ValueError(f'No {array} antenna pattern implemented for N={N} elements')
        
    def plot_antenna_pattern(self, block=True):
        ''' 
        Plotting antenna pattern.
        '''
        from matplotlib import pyplot as plt
        plt.figure()
        plt.scatter(self.ant_pos[:,0], self.ant_pos[:,1])
        ax = plt.gca()
        for ii in range(self.ant_pos.shape[0]):
            ax.text(self.ant_pos[ii,0], self.ant_pos[ii,1], f'{ii}')
        plt.show(block=block)
    
    def get_steering_vectors(self, angles: np.ndarray):
        '''
        Compute steering vectors for given angles of incidence in radians.

        Arguments
        ---------
        angles : 2d array
            numpy array of shape (nsig, 2) containing angles of arrival in the order (az, el) in radians

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
    def get_random_angles(num_angles: int):
        '''
        Generate random angles of arrival (azimuth, elevation) of shape (num_angles, 2) in radians.
        '''
        az = np.random.rand(num_angles) * 2.0*np.pi
        el = 15/360*np.random.randn(num_angles) * 2.0*np.pi #TODO?
        return np.stack([az, el], axis=-1)

if __name__ == '__main__':
    Nant = 4
    ar = ArrayReceiver(Nant, 'UCA')
    ar.plot_antenna_pattern()
    sv = ar.get_steering_vectors(np.array([[0,0],[np.pi/4,0],[np.pi/2,0]]))
    print(sv)