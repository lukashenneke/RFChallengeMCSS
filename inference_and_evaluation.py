import os, sys
import numpy as np
import h5py
from tqdm import tqdm
from scipy.signal import find_peaks

import torch
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import rfcutils
from omegaconf import OmegaConf
from src.config_torchwavenet import Config, parse_configs
from src.torchwavenet import Wave
from src.dataset import ArrayReceiver

get_db = lambda p: 10*np.log10(p)
get_pow = lambda s: np.mean(np.abs(s)**2, axis=-1)
get_sinr = lambda s, i: get_pow(s)/get_pow(i)
get_sinr_db = lambda s, i: get_db(get_sinr(s,i))

sig_len = 40960
n_per_batch = 100
all_sinr = np.arange(-30, 0.1, 3)

seed_number = 0

def get_soi_generation_fn(soi_sig_type):
    if soi_sig_type == 'QPSK':
        generate_soi = lambda n, s_len: rfcutils.generate_qpsk_signal(n, s_len//16)
        demod_soi = rfcutils.qpsk_matched_filter_demod
    elif soi_sig_type == 'QAM16':
        generate_soi = lambda n, s_len: rfcutils.generate_qam16_signal(n, s_len//16)
        demod_soi = rfcutils.qam16_matched_filter_demod
    elif soi_sig_type ==  'QPSK2':
        generate_soi = lambda n, s_len: rfcutils.generate_qpsk2_signal(n, s_len//4)
        demod_soi = rfcutils.qpsk2_matched_filter_demod
    elif soi_sig_type == 'OFDMQPSK':
        generate_soi = lambda n, s_len: rfcutils.generate_ofdm_signal(n, s_len//80)
        _,_,_,RES_GRID = rfcutils.generate_ofdm_signal(1, sig_len//80)
        demod_soi = lambda s: rfcutils.ofdm_demod(s, RES_GRID)
    else:
        raise Exception("SOI Type not recognized")
    return generate_soi, demod_soi

def load_model(cfg_file, model_file):
    cfg = OmegaConf.load(cfg_file)
    cfg: Config = Config(**parse_configs(cfg, None))
    cfg.model.input_channels = 2*cfg.data.num_ant
    nn_model = Wave(cfg.model).cuda()
    nn_model.load_state_dict(torch.load(model_file)['model'])
    return nn_model, cfg.data.num_ant, cfg.data.array_type
    
def get_array_from_name(id_string):
    if 'ULA' in id_string.upper():
        array_type = 'ULA'
    elif 'UCA' in id_string.upper():
        array_type = 'UCA'
    elif 'URA' in id_string.upper():
        array_type = 'URA'
    else:
        array_type = 'ULA'
    splt = id_string.split('_')
    for s in splt:
        if s[-2:] == 'ch':
            try:
                n_channels = int(s[:-2])
                break
            except:
                pass
    return n_channels, array_type

def get_mixtures(n_channels, array_type, soi, intf, ang, noise):
    # multi channel signal mixtures
    arr = ArrayReceiver(n_channels, array_type)
    steer_vec = arr.get_steering_vectors(ang.reshape((-1,2))).T
    steer_vec = steer_vec.reshape((-1,2,n_channels)).transpose((0,2,1))
    x = np.stack([soi, intf], axis=1)
    y = np.matmul(steer_vec, x)
    y += noise[:, :n_channels]
    return y

def get_mixtures_bf(n_channels, array_type, soi, intf, ang, noise, id_string):
    # mix signals and apply null steering beamformer
    arr = ArrayReceiver(n_channels, array_type)
    steer_vec = arr.get_steering_vectors(ang.reshape((-1,2))).T
    steer_vec = steer_vec.reshape((-1,2,n_channels)).transpose((0,2,1))
    x = np.stack([soi, intf], axis=1)
    y = np.matmul(steer_vec, x)
    y += noise[:, :n_channels]
    
    if 'oracle' in id_string.lower(): # Oracle == exact AoA / steering vectors
        pass
    elif 'music' in id_string.lower(): # MUSIC AoA estimation
        res_fac = 1
        az = np.arange(360*res_fac)/360 * 2*np.pi
        el = np.arange(50*res_fac)/360 * 2*np.pi
        angles = np.stack([np.tile(az, el.shape[0]), np.repeat(el, az.shape[0])], axis=1)
        #angles = np.stack([az, np.zeros(az.shape)], axis=1)
        sv = arr.get_steering_vectors(angles).T
        
        cov = np.matmul(y, y.transpose((0,2,1)).conj()) / y.shape[2]
        U, S, VH = np.linalg.svd(cov, hermitian=True)
        #P = 1/np.sum(np.abs(sv.conj()[None,:,:,None] * U[:,None,:,2:])**2, axis=(2,3))
        w, v = np.linalg.eig(cov)
        eig_val_order = np.argsort(np.abs(w), axis=-1) # find order of magnitude of eigenvalues
        vs = np.zeros(v.shape, dtype=v.dtype)
        for i, o in enumerate(eig_val_order):
            for j, oo in enumerate(o[::-1]):
                vs[i,:,j] = v[i,:,oo]
        vs = vs[:,:,2:]
        vvh = np.matmul(vs, vs.conj().transpose((0,2,1)))

        P = np.log10(np.abs(1 / np.matmul(sv.conj()[None,:,None,:], np.matmul(vvh[:,None], sv[None,:,:,None]))).squeeze())
        for i, p in enumerate(P):
            ind, _ = find_peaks(p)
        from matplotlib import pyplot as plt
        plt.plot(P[-1])
        plt.show(block=True)
        raise NotImplementedError()
    elif 'err' in id_string.lower(): # Random estimation error in AoA
        sigma = 1 # degree
        err = np.random.randn(*ang.shape) * sigma * 2*np.pi/360
        steer_vec = arr.get_steering_vectors((ang+err).reshape((-1,2))).T
        steer_vec = steer_vec.reshape((-1,2,n_channels)).transpose((0,2,1))
    else:
        raise ValueError(f'Partly unknown id_string {id_string}')
    if 'mvdr' in id_string.lower(): # MVDR
        rn = np.matmul(steer_vec[:,:,1:2], np.stack([intf], axis=1)) + noise[:, :n_channels]
        cov_inv = np.linalg.pinv(np.matmul(rn, rn.transpose((0,2,1)).conj()) / rn.shape[2])
        W = np.matmul(cov_inv, steer_vec).conj()
        scale = np.matmul(steer_vec.transpose((0,2,1)), W)
    elif 'mpdr' in id_string.lower(): # MPDR
        cov_inv = np.linalg.pinv(np.matmul(y, y.transpose((0,2,1)).conj()) / y.shape[2]) # inverse covariance matrice 
        W = np.matmul(cov_inv, steer_vec).conj()
        scale = np.matmul(steer_vec.transpose((0,2,1)), W)
    elif 'ns' in id_string.lower(): # Null steering 
        W = np.matmul(steer_vec.conj(), np.linalg.pinv(np.matmul(steer_vec.transpose((0,2,1)), steer_vec.conj())))
    else:
        raise ValueError(f'Partly unknown id_string {id_string}')
    bf = np.matmul(W.transpose((0,2,1)), y)
    #result = _projection(soi, bf[:,0])
    result = bf[:,0] #/ scale[:,0:1,0] # without beamformer scaling for better MSE scores 
    #resid = soi - result
    return result[:,None]

def _projection(x, s):
    num = np.sum(x.conj() * s, axis=1)
    denom = np.sum(s.conj() * s, axis=1)
    alpha = num/denom
    res = alpha[:,None] * s
    return res

def run_inference(all_sig_mixture, soi_type, nn_model):
    # inference pipeline
    generate_soi, demod_soi = get_soi_generation_fn(soi_type)

    if nn_model is not None:
        with torch.no_grad():
            nn_model.eval()
            all_sig1_out = []
            bsz = 100
            for i in tqdm(range(all_sig_mixture.shape[0]//bsz), leave=False):
                sig_input = torch.from_numpy(all_sig_mixture[i*bsz:(i+1)*bsz])
                sig_input = torch.view_as_real(sig_input).transpose(-2,-1)
                sig_input = torch.flatten(sig_input, start_dim=1, end_dim=2).to('cuda')
                sig1_out = nn_model(sig_input)
                all_sig1_out.append(sig1_out.transpose(1,2).detach().cpu().numpy())
    else: # direct demodulation of all_sig_mixture (1st channel)
        all_sig1_out = np.stack([all_sig_mixture[:,0,:].real, all_sig_mixture[:,0,:].imag], axis=-1)
    sig1_out = tf.concat(all_sig1_out, axis=0)
    sig1_est = tf.complex(sig1_out[:,:,0], sig1_out[:,:,1])

    bit_est = []
    for idx, sinr_db in tqdm(enumerate(all_sinr), leave=False):
        bit_est_batch, _ = demod_soi(sig1_est[idx*n_per_batch:(idx+1)*n_per_batch])
        bit_est.append(bit_est_batch)
    bit_est = tf.concat(bit_est, axis=0)
    sig1_est, bit_est = sig1_est.numpy(), bit_est.numpy()
    return sig1_est, bit_est

def run_demod_test(sig1_est, bit1_est, all_sig1, all_bits1):    
    # demod pipeline
    def eval_mse(all_sig_est, all_sig_soi):
        assert all_sig_est.shape == all_sig_soi.shape, 'Invalid SOI estimate shape'
        return np.mean(np.abs(all_sig_est - all_sig_soi)**2, axis=1)
    
    def eval_ber(bit_est, bit_true):
        ber = np.sum((bit_est != bit_true).astype(np.float32), axis=1) / bit_true.shape[1]
        assert bit_est.shape == bit_true.shape, 'Invalid bit estimate shape'
        return ber

    all_mse, all_ber = [], [] 
    for idx, sinr in enumerate(all_sinr):
        batch_mse =  eval_mse(sig1_est[idx*n_per_batch:(idx+1)*n_per_batch], all_sig1[idx*n_per_batch:(idx+1)*n_per_batch])
        bit_true_batch = all_bits1[idx*n_per_batch:(idx+1)*n_per_batch]
        batch_ber = eval_ber(bit1_est[idx*n_per_batch:(idx+1)*n_per_batch], bit_true_batch)
        all_mse.append(batch_mse)
        all_ber.append(batch_ber)

    all_mse, all_ber = np.array(all_mse), np.array(all_ber)
    mse_mean = 10*np.log10(np.mean(all_mse, axis=-1))
    ber_mean = np.mean(all_ber, axis=-1)
    return mse_mean, ber_mean

def main(soi_type, interference_sig_type, id_string, testset_identifier):

    # load evaluation data
    with h5py.File(os.path.join('dataset', f'{testset_identifier}_Dataset_{soi_type}_{interference_sig_type}.h5'), 'r') as hf:
        intf = np.array(hf.get('interferences'))
        soi = np.array(hf.get('soi'))
        noise = np.array(hf.get('noise'))
        ang = np.array(hf.get('angles'))
        all_sig1 = np.array(hf.get('soi'))
        all_bits1 = np.array(hf.get('bits'))

    # load model
    if id_string.startswith('bf'):
        if 'wavenet' in id_string:
            nn_model, _, _ = load_model(f'src/configs/wavenet.yml', os.path.join('models', f'{soi_type}_{interference_sig_type}_wavenet', 'weights.pt'))
        else:
            nn_model = None
        n_channels, array_type = get_array_from_name(id_string)
    elif id_string == 'none':
        nn_model, n_channels, array_type = None, 1, 'ULA'
    else:
        nn_model, n_channels, array_type = load_model(f'src/configs/{id_string}.yml', os.path.join('models', f'{soi_type}_{interference_sig_type}_{id_string}', 'weights.pt'))

    # inference on evaluation data
    mse_ber = []
    np.random.seed(seed_number)
    for i in range(ang.shape[0] if n_channels>1 else 1):
        print('Subset', i)
        if id_string.startswith('bf'):
            all_sig_mixture = get_mixtures_bf(n_channels, array_type, soi, intf, ang[i], noise, id_string)
        else:
            all_sig_mixture = get_mixtures(n_channels, array_type, soi, intf, ang[i], noise)
        sig1_est, bit1_est = run_inference(all_sig_mixture, soi_type, nn_model)
        mse_mean, ber_mean = run_demod_test(sig1_est, bit1_est, all_sig1, all_bits1)
        mse_ber.append(np.stack([mse_mean, ber_mean], axis=0))

    # save results
    mse_ber = np.mean(np.stack(mse_ber, axis=-1), axis=-1)
    np.save(os.path.join('outputs', f'{id_string}_{testset_identifier}_{soi_type}_{interference_sig_type}_results'), mse_ber)

if __name__ == "__main__":
    # input
    id_string = [sys.argv[1]] if len(sys.argv) > 1 else ['none', 'bf_mpdr_oracle_2ch_ULA', 'bf_mpdr_oracle_4ch_ULA', 'bf_mpdr_oracle_4ch_URA', 'wavenet', 'wavenet_2ch', 'wavenet_4ch', 'wavenet_4ch_URA']
    soi_type = [sys.argv[3]] if len(sys.argv) > 3 else ['QPSK', 'OFDMQPSK']
    interference_sig_type = [sys.argv[4]] if len(sys.argv) > 4 else ['EMISignal1', 'CommSignal2', 'CommSignal3', 'CommSignal5G1']
    testset_identifier = sys.argv[5] if len(sys.argv) > 5 else 'TestSet1Example'

    # call main
    for ids in id_string:
        for s in soi_type:
            for i in interference_sig_type:
                try:
                    print(ids, s, i)
                    main(s, i, ids, testset_identifier)
                except Exception as e:
                    print(e)
