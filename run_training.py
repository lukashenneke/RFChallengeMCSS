import sys
from train import main

soi = ['QPSK', 'OFDMQPSK']
interference = ['EMISignal1', 'CommSignal2', 'CommSignal3', 'CommSignal5G1']
cfg = ['wavenet', 'wavenet_2ch', 'wavenet_4ch', 'wavenet_4ch_URA']

if __name__ == '__main__':
    tmp = sys.argv
    for i in interference:
        for s in soi:
            for c in cfg:
                print(s, i, c)
                sys.argv = tmp + [s, i, '-id', c]
                try:
                    main()
                except SystemExit as e:
                    pass
                except RuntimeError as r:
                    print(' ')
                    print(r)
                    pass