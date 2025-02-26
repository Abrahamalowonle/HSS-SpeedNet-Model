# HSS-SpeedNet-Model

![Project Banner](./aia_map.png)

> Prediction of High-Speed Solar Wind Streams using SDO/AIA EUV MAPs/BINARY Maps.
This is a follow up resource for our Research Publication.
---

## Table of Contents

- [About SpeedNet Model](#about-SpeedNet-Model)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## About SpeedNet Model

The structure of the sun and it ability to influence the propagation of space plasma parcel, realised that different dynamo effect in the sun leads to the generation of solar wind in the interplanetary. The presence of CME (Coronal Mass Ejection) would lead to an abrupt change in the magnitude and effect of solar wind parcel in the interplanetary and near earth. Also, the presence of Coronal hole could create the CIR (Corotating Interaction region) and the SIR (Stream Interaction region), which distort and changes the flow, direction and the magnitude of the Solar wind parcel. This study therefore enacts the prediction of solar wind speed not associated with ICMEs (Interplanetary Coronal Mass Ejection) using Convolution Neural Network (CNN) during the different phases of the solar cycle (SC 24 and 25). 

Solutions:

- Mapping Correction/Offlimb-removal.
- ICME Eliminnation and Map adaptation based on delays and folds.
- Prediction of High Speed Streams(HSSs).
- Quantitative Evaluation of the Model Performance.
- Grad-Cam Observation of the model performance.

![Project Banner](./Image_processes.png)
---

## Installation

To set up the environment and install dependencies from the `environment.yml` file, follow these steps:

```bash
# Clone the repository
git clone https://github.com/Abrahamalowonle/HSS-SpeedNet-Model.git

# Navigate to the project directory
cd HSS-SpeedNet-Model


```
### **1. Install Conda (If Not Already Installed)**
If you haven't installed Conda, download and install **[Miniconda](https://docs.conda.io/en/latest/miniconda.html)** or **[Anaconda](https://www.anaconda.com/)**.

For WSL users, install Miniconda/Anaconda using:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

```
### **2. Create a New Environment from `environment.yml`**
Run the following command to create a new Conda environment:

```bash
conda env create -f environment.yml

````
---

## Usage

How to use the project:

1. Process the Solar wind speed using SW_Processing.
2. Process the Maps using Image_Preprocess.
3. Perform the Analysis:
    - a. SpeedNet_EUV for the Three Channel EUV Model.
    - b. SpeedNet_BM for the Binary Map Model.
4. HSEMETRIC contains metrics for analyzing the Threat Score.
5. To observe the Activation of the Model, utilize HSS ACtivation metrics.
6. For Visualizing the Model Plot:
    - HSS_Activation - HSSs_Activation_comparison_Analysis.
    - Timeseries and GradCam for SpeedNet_BM and SpeedNet_EUV.
---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
