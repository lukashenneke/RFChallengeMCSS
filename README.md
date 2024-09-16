# Multi-channel extension of the RF signal separation challenge
Accompanying code for the paper [Extending the Single-Channel RF Signal Separation Challenge to Multi-Antenna Scenarios](#reference).

The paper is based on the [ICASSP 2024 SP Grand Challenge: Data-Driven Signal Separation in Radio Spectrum](https://signalprocessingsociety.org/publications-resources/data-challenges/data-driven-signal-separation-radio-spectrum-icassp-2024).

## Challenge
Click [here](https://rfchallenge.mit.edu/icassp24-single-channel/) for details on the challenge setup.
This GitHub repository is a fork of the challenge organizers' GitHub repository providing the [starter code](https://github.com/RFChallenge/icassp2024rfchallenge).
The RF challenge data can be downloaded manually here: [InterferenceSet](https://www.dropbox.com/scl/fi/zlvgxlhp8het8j8swchgg/dataset.zip?rlkey=4rrm2eyvjgi155ceg8gxb5fc4&dl=0).

## Setup
The code is only tested using Python 3.8.5 and the package versions listed in `requirements.txt`.
Relevant bash commands to set up the code:
```bash
# clone this repository
git clone https://github.com/lukashenneke/RFChallengeSCSS.git
cd RFChallengeSCSS

# install python packages - using Python 3.8 and a virtual environment is recommended to make things work
python install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116

# obtain the dataset
wget -O dataset.zip "https://www.dropbox.com/scl/fi/zlvgxlhp8het8j8swchgg/dataset.zip?rlkey=4rrm2eyvjgi155ceg8gxb5fc4&dl=0"
unzip dataset.zip
rm dataset.zip
```

## Training

The interface for training a signal separation model for a combination of SOI Type and Interference Type is

```bash
# start training using the config file name as identifier, 
# e.g. 'wavenet_2ch' to refer to src/configs/wavenet_2ch.yml
python train.py [SOI Type] [Interference Type] -id [YAML identifier]
```
In addition to the training settings, the antenna array geometry and the number of channels are also stored in the YAML config file.
The config files for all antenna arrays studied in [Extending the Single-Channel RF Signal Separation Challenge to Multi-Antenna Scenarios](#reference) are placed in [src/configs](src/configs).

To re-run all experiments (this will probably take weeks), run:

```bash
python run_training.py
```

To only run a subset of the experiments, delete unwanted entries in [run_training.py](run_training.py).

## Inference and evaluation

Before starting inference, the evaluation datasets "TestSet1Example", based on the raw interference dataset of TestSet1, have to be generated for all signal mixture scenarios:
```bash
# generate TestSet1Example for specific signal mixture scenario
python testmixture_generator.py [SOI Type] [Interference Type]

# generate TestSet1Example for all 8 scenarios
python testmixture_generator.py
```

To evaluate trained models and beamforming approaches, run:
```bash
# evaluate model for specific signal mixture scenario
python inference_and_evaluation.py [Method Identifier] [SOI Type] [Interference Type]

# evaluate model for all 8 scenarios
python inference_and_evaluation.py [Method Identifier]
```

This will save evaluation results to a *.npy file in [outputs](outputs/).
Here, SOI demodulation without interference mitigation can be started with `[Method Identifier] = 'none'`.
In order to use MPDR beamformers for signal separation utilizing exact steering vectors, use e.g. `[Method Identifier] = 'bf_mpdr_oracle_2ch_ULA'` for beamforming based on a 2-element Uniform Linear Array and `[Method Identifier] = 'bf_mpdr_oracle_4ch_URA'` for a 4-element Uniform Rectangular Array.
For evaluation of all methods and signal scenarios considered in the paper (assuming all WaveNet models have been trained), solely run
```bash
python inference_and_evaluation.py
```

Finally, run 

```bash
python plot_results.py [SOI Type] [Interference Type]
```

to plot and print the results. It might be necessary to add your Method Identifier to the methods list in [plot_results.py](./plot_results.py).

## Reference
COMING SOON