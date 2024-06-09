# Starter Code for ICASSP 2024 SP Grand Challenge: Data-Driven Signal Separation in Radio Spectrum

[Click here for details on the challenge setup](https://rfchallenge.mit.edu/wp-content/uploads/2023/09/ICASSP24_single_channel.pdf)


## About this Repository
For those eager to dive in, we have prepared a concise guide to get you started.

Check out [notebook/RFC_QuickStart_Guide.ipynb](https://github.com/RFChallenge/icassp2024rfchallenge/blob/0.2.0/notebook/RFC_QuickStart_Guide.ipynb) for practical code snippets. You will find steps to create a small but representative training set and steps for inference to generate your submission outputs.
For a broader understanding and other helpful resources in the starter kit integral to the competition, please see the details and references provided below.

[Link to InterferenceSet](https://www.dropbox.com/scl/fi/zlvgxlhp8het8j8swchgg/dataset.zip?rlkey=4rrm2eyvjgi155ceg8gxb5fc4&dl=0)

## TestSet for Evaluation

This starter kit equips you with essential resources to develop signal separation and interference rejection solutions. In this competition, the crux of the evaluation hinges on your ability to handle provided signal mixtures. Your task will be twofold:

1.  Estimate the Signal of Interest (SOI) component within the two-component mixture.

2.  Deduce the best possible estimate of the underlying information bits encapsulated in the SOI.

Delve into the specifics below for comprehensive details.




### Submission Specifications:

For every configuration defined by a specific SOI Type and Interference Type, participants are required to provide:

1.  SOI Component Estimate:
-   A numpy array of dimensions 1,100 x 40,960.
-   This should contain complex values representing the estimated SOI component present.
-   Filename: `[ID String]_[TestSet Identifier]_estimated_soi_[SOI Type]_[Interference Type].npy`
    (where ID String will be a unique identifier, e.g., your team name)

2.  Information Bits Estimate:
-   A numpy array of dimensions 1,100 x B.    
-   The value of B depends on the SOI type:
    -   B = 5,120 for QPSK SOI
    -   B = 57,344 for OFDMQPSK SOI
-   The array should exclusively contain values of 1’s and 0’s, corresponding to the estimated information bits carried by the SOI.
-   Filename: `[ID String]_[TestSet Identifier]_estimated_bits_[SOI Type]_[Interference Type].npy`
    (where ID String will be a unique identifier, e.g., your team name)

For guidance on mapping the SOI signal to the information bits, participants are advised to consult the provided demodulation helper functions (e.g., as used in [notebook/RFC_EvalSet_Demo.ipynb](https://github.com/RFChallenge/rfchallenge_singlechannel_starter_grandchallenge2023/blob/0.2.0/notebook/RFC_EvalSet_Demo.ipynb)).

Submissions should be sent to the organizers at rfchallenge@mit.edu.

_The intellectual property (IP) is not transferred to the challenge organizers; in other words, if code is shared or submitted, the participants retain ownership of their code._

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

For a complete overview of the dependencies within our Anaconda environment, please refer [here (rfsionna)](https://github.com/RFChallenge/icassp2024rfchallenge/blob/0.2.0/rfsionna_env.yml). Additionally, if you're interested in the PyTorch-based baseline, you can find the respective Anaconda environment dependencies that the organizers used [here (rftorch)](https://github.com/RFChallenge/icassp2024rfchallenge/blob/0.2.0/rftorch_env.yml).

Since participants are tasked with running their own inference, we are currently not imposing restrictions on the libraries for training and inference. However, the submissions are expected to be in the form of numpy arrays (`.npy` files) that are compatible with our system (`numpy==1.21.6`).

> Note: Diverging from the versions of the dependencies listed above might result in varied behaviors of the starter code. Participants are advised to check for version compatibility in their implementations and solutions.


## Helper Functions for Testing:

To assist participants during testing, we provide several example scripts designed to create and test with evaluation sets analogous to TestSet1Mixture.

`python sampletest_testmixture_generator.py [SOI Type] [Interference Type]`

This script generates a new evaluation set (default name: TestSet1Example) based on the raw interference dataset of TestSet1. Participants can employ this for cross-checking. The produced outputs include a mixture numpy array, a metadata numpy array (similar to what's given in TestSet1Mixture), and a ground truth file. Participants can also change the seed number to generate new instances of such example test sets.

(An example generated, named TestSet1Example (using seed_number=0), can be found [here](https://drive.google.com/file/d/1D1rHwEBpDRBVWhBGalEGJ0OzYbBeb4il/view?usp=drive_link).)


`python sampletest_tf_unet_inference.py [SOI Type] [Interference Type] [TestSet Identifier]`

`python sampletest_torch_wavenet_inference.py [SOI Type] [Interference Type] [TestSet Identifier]`

(Default: Use TestSet1Example for [TestSet Identifier])
Scripts that leverage the supplied baseline methods (Modified U-Net on Tensorflow or WaveNet on PyTorch) for inference.

`python sampletest_evaluationscript.py [SOI Type] [Interference Type] [TestSet Identifier] [Method ID String]`

[Method ID String] is your submission's unique identifier---refer to submission specifications.
Utilize this script to assess the outputs generated from the inference script.


## Helper Functions for Training:


    
2.  Model Training Scripts: The competition organizers have curated two implementations:
    -   UNet on Tensorflow: `train_unet_model.py`, accompanied with neural network specification in `src/unet_model.py`
    -   WaveNet on Torch: `train_torchwavenet.py`, accompanied with dependencies including `supervised_config.yml` and `src/configs`, `src/torchdataset.py`, `src/learner_torchwavenet.py`, `src/config_torchwavenet.py` and `src/torchwavenet.py`  

