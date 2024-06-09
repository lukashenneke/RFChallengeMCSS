import os, sys, h5py
import numpy as np
import matplotlib.pyplot as plt

all_sinr = np.arange(-30, 0.1, 3)
n_per_batch = 100

def run_demod_test(sig1_est, bit1_est, soi_type, interference_sig_type, testset_identifier):
    # For SampleEvalSet
    with h5py.File(os.path.join('dataset', f'{testset_identifier}_Dataset_{soi_type}_{interference_sig_type}.h5'), 'r') as hf:
        all_sig1 = np.array(hf.get('soi'))
        all_bits1 = np.array(hf.get('bits'))
    
    # Evaluation pipeline
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

def run_evaluation(testset_identifier, soi_identifier, interference_identifier):
    if isinstance(soi_identifier, str): soi_identifier = [soi_identifier]
    if isinstance(interference_identifier, str): interference_identifier = [interference_identifier]
    
    keep_scores = {}
    for soi_type in soi_identifier:
        for interference_sig_type in interference_identifier:
            all_mse, all_ber, all_scores = {}, {}, {}
            for id_string in ['default', 'wavenet', 'soi-ae']:
                for nch in [1, 2, 4]:
                    try:
                        sig1_est = np.load(os.path.join('outputs', f'{id_string}_{testset_identifier}_estimated_soi_{soi_type}_{interference_sig_type}_{nch}ch.npy'))
                    except FileNotFoundError:
                        continue
                    bit1_est = np.load(os.path.join('outputs', f'{id_string}_{testset_identifier}_estimated_bits_{soi_type}_{interference_sig_type}_{nch}ch.npy')) # _output_bits_ _estimated_bits_
                    
                    mse_mean, ber_mean = run_demod_test(sig1_est, bit1_est, soi_type, interference_sig_type, testset_identifier)
                    
                    id_string_ch = f'{id_string}_{nch}ch'
                    all_mse[id_string_ch] = mse_mean
                    all_ber[id_string_ch] = ber_mean
                    mse_score = mse_mean.copy()
                    mse_score[mse_score<-50] = -50
                    mse_score = np.mean(mse_score)
                    ber_score = -(sum(ber_mean < 1e-2)-1)*3
                    all_scores[id_string_ch] = (mse_score, ber_score)

            if len(all_scores) == 0: continue
            comb = f'{soi_type}_{interference_sig_type}'
            keep_scores[comb] = all_scores
            print('===', comb, '===')
            for k,v in all_scores.items(): print(k, v)

            tr_dict = {'abc': 'Proposed_Model', 'cdf': 'Proposed_Model_FT'}
            plt.figure()
            for id_string in all_mse.keys():
                plt.plot(all_sinr, all_mse[id_string], 'x--', label=tr_dict.get(id_string, id_string))
            plt.legend()
            plt.grid()
            plt.gca().set_ylim(top=3)
            plt.xlabel('SINR [dB]')
            plt.ylabel('MSE [dB]')
            plt.title(f'MSE - {soi_type} + {interference_sig_type}')
            plt.show(block=False)

            plt.figure()
            for id_string in all_ber.keys():
                plt.semilogy(all_sinr, all_ber[id_string], 'x--', label=tr_dict.get(id_string, id_string))
            plt.legend()
            plt.grid()
            plt.ylim([1e-4, 1])
            plt.xlabel('SINR [dB]')
            plt.ylabel('BER')
            plt.title(f'BER - {soi_type} + {interference_sig_type}')
            plt.show(block=True)
    final_score = {}
    for k,v in keep_scores.items():
        for kk,vv in v.items():
            if kk not in final_score:
                final_score[kk] = vv
            else:
                final_score[kk] = (final_score[kk][0]+vv[0], final_score[kk][1]+vv[1])
    print('=== FINAL SCORES ===')
    for k,v in final_score.items(): print(k, v)

if __name__ == "__main__":
    soi_type = sys.argv[1] if len(sys.argv) > 1 else ['QPSK', 'OFDMQPSK']
    interference_sig_type = sys.argv[2] if len(sys.argv) > 2 else ['EMISignal1', 'CommSignal2', 'CommSignal3', 'CommSignal5G1']
    testset_identifier = sys.argv[3] if len(sys.argv) > 3 else 'TestSet1Example'

    run_evaluation(testset_identifier, soi_type, interference_sig_type)