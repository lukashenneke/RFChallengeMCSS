# Multi Channel Signal Separation
# Extension of the ICASSP 2024 RF Challenge for Single Channel Signal Separation

Add github repo, challenge paper, challenge website, description pdf
[Click here for details on the challenge setup](https://rfchallenge.mit.edu/icassp24-single-channel/)

## RF Challenge Data
Refer to website/paper/...

### InterferenceSet:
[Link to InterferenceSet](https://www.dropbox.com/scl/fi/zlvgxlhp8het8j8swchgg/dataset.zip?rlkey=4rrm2eyvjgi155ceg8gxb5fc4&dl=0)

### TestSet1:
[Click here for TestSet1Mixture files](https://www.dropbox.com/scl/fi/d2kjtfmbh3mgxddbubf80/TestSet1Mixture.zip?rlkey=lwhzt1ayn2bqwosc9o9cq9dwr&dl=0)

50 frames of each interference type have been reserved to form TestSet1 (interference frames). These will be released alongside the main dataset (InterferenceSet frames), and the mixtures from TestSet1Mixture are generated from this collection. Please note that although TestSet1 is available for examination, the final evaluation for participants will be based on a hidden, unreleased set (TestSet2 interference frames).

### TestSet2:
[Click here for TestSet2Mixture files](https://www.dropbox.com/scl/fi/m36l2imiit5svqz1yz46g/TestSet2Mixture.zip?rlkey=n5mwzi11l55l2xzfw9ee5m0ye&dl=0)

## Starter Code Setup:
Relevant bash commands to set up the starter code:
```bash
git clone https://github.com/RFChallenge/icassp2024rfchallenge.git rfchallenge
cd rfchallenge

# To obtain the dataset
wget -O  dataset.zip "https://www.dropbox.com/scl/fi/zlvgxlhp8het8j8swchgg/dataset.zip?rlkey=4rrm2eyvjgi155ceg8gxb5fc4&dl=0"
unzip  dataset.zip
rm dataset.zip

# To obtain TestSet1Mixture
wget -O  TestSet1Mixture.zip  "https://www.dropbox.com/scl/fi/d2kjtfmbh3mgxddbubf80/TestSet1Mixture.zip?rlkey=lwhzt1ayn2bqwosc9o9cq9dwr&dl=0"
unzip TestSet1Mixture.zip -d dataset
rm TestSet1Mixture.zip
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

### Acknowledgements
The efforts of the organizers are supported by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government.

The organizers acknowledge the MIT SuperCloud and Lincoln Laboratory Supercomputing Center for providing HPC resources that have contributed to the development of this work.
