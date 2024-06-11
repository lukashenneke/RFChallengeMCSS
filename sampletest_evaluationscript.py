import os, sys
import numpy as np
import matplotlib.pyplot as plt

all_sinr = np.arange(-30, 0.1, 3)
n_per_batch = 100

def run_evaluation(testset_identifier, soi_identifier, interference_identifier):
    keep_scores = {}
    for soi_type in soi_identifier:
        for interference_sig_type in interference_identifier:
            all_mse, all_ber, all_scores = {}, {}, {}
            for id_string in ['wavenet', 'bf', 'bf+wavenet']: # 'baseline', 'bf-oracle', 'soi-ae', 
                for nch in [1, 2, 4]:
                    try:
                        results = np.load(os.path.join('outputs', f'{id_string}_{testset_identifier}_{soi_type}_{interference_sig_type}_{nch}ch_results.npy'))
                    except FileNotFoundError:
                        continue
                    mse_mean, ber_mean = results[0], results[1]
                    
                    id_string_ch = f'{id_string}_{nch}ch'
                    all_mse[id_string_ch] = mse_mean
                    all_ber[id_string_ch] = ber_mean
                    mse_score = mse_mean.copy()
                    mse_score[mse_score<-50] = -50
                    mse_score = np.mean(mse_score)
                    ber_score = -(sum(ber_mean < 1e-2)-1)*3
                    all_scores[id_string_ch] = (mse_score, ber_score)

            if len(all_scores) <= 1: continue
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
    soi_type = [sys.argv[1]] if len(sys.argv) > 1 else ['QPSK', 'OFDMQPSK']
    interference_sig_type = [sys.argv[2]] if len(sys.argv) > 2 else ['EMISignal1', 'CommSignal2', 'CommSignal3', 'CommSignal5G1']
    testset_identifier = sys.argv[3] if len(sys.argv) > 3 else 'TestSet1Example'

    run_evaluation(testset_identifier, soi_type, interference_sig_type)