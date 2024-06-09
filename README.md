# Multi Channel Signal Separation
# Extension of the ICASSP 2024 RF Challenge for Single Channel Signal Separation

Add github repo, challenge paper, challenge website, description pdf
[Click here for details on the challenge setup](https://rfchallenge.mit.edu/icassp24-single-channel/)

## RF Challenge Data
Refer to website/paper/...

### InterferenceSet:
[Link to InterferenceSet](https://www.dropbox.com/scl/fi/zlvgxlhp8het8j8swchgg/dataset.zip?rlkey=4rrm2eyvjgi155ceg8gxb5fc4&dl=0)


## Task

This starter kit equips you with essential resources to develop signal separation and interference rejection solutions. In this competition, the crux of the evaluation hinges on your ability to handle provided signal mixtures. Your task will be twofold:

1.  Estimate the Signal of Interest (SOI) component within the two-component mixture.

2.  Deduce the best possible estimate of the underlying information bits encapsulated in the SOI.

Delve into the specifics below for comprehensive details.


## Starter Code Setup:
Relevant bash commands to set up the starter code:
```bash
git clone https://github.com/lukashenneke/RFChallengeMCSS.git
cd RFChallengeMCSS

# To obtain the dataset
wget -O  dataset.zip "https://www.dropbox.com/scl/fi/zlvgxlhp8het8j8swchgg/dataset.zip?rlkey=4rrm2eyvjgi155ceg8gxb5fc4&dl=0"
unzip  dataset.zip
rm dataset.zip
```

Dependencies: The organizers have used the following libraries to generate the signal mixtures and test the relevant baseline models
* python==3.7.13
* numpy==1.21.6
* tensorflow==2.8.2
* sionna==0.10.0
* tqdm==4.64.0
* h5py==3.7.0




## Helper Functions for Testing:

To assist participants during testing, we provide several example scripts designed to create and test with evaluation sets analogous to TestSet1Mixture.

`python sampletest_testmixture_generator.py [SOI Type] [Interference Type]`

This script generates a new evaluation set (default name: TestSet1Example) based on the raw interference dataset of TestSet1. Participants can employ this for cross-checking. The produced outputs include a mixture numpy array, a metadata numpy array (similar to what's given in TestSet1Mixture), and a ground truth file. Participants can also change the seed number to generate new instances of such example test sets.

`python sampletest_torch_wavenet_inference.py [SOI Type] [Interference Type] [TestSet Identifier]`

(Default: Use TestSet1Example for [TestSet Identifier])
Scripts that leverage the supplied baseline methods (Modified U-Net on Tensorflow or WaveNet on PyTorch) for inference.

`python sampletest_evaluationscript.py [SOI Type] [Interference Type] [TestSet Identifier] [Method ID String]`

[Method ID String] is your submission's unique identifier---refer to submission specifications.
Utilize this script to assess the outputs generated from the inference script.

    
2.  Model Training Scripts: The competition organizers have curated two implementations:
    -   UNet on Tensorflow: `train_unet_model.py`, accompanied with neural network specification in `src/unet_model.py`
    -   WaveNet on Torch: `train_torchwavenet.py`, accompanied with dependencies including `supervised_config.yml` and `src/configs`, `src/torchdataset.py`, `src/learner_torchwavenet.py`, `src/config_torchwavenet.py` and `src/torchwavenet.py`