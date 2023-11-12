

## 1. What is scHiC-SSM?

Single-cell Hi-C (scHi-C) profiles the chromatin three-dimensional (3D) conformation at single cell resolution, thus revealing genomic interactions with potential impact in regulating cell identities. However, high dimensionality and sparsity of scHi-C data often complicate the analysis. Here, we introduce a method, scHiC-SSM, for embedding scHi-C data that combines a semi-supervised deep generative model to learn latent features that accurately characterize scHi-C data. scHiC-SSM outperforms other tools in typical aspects of scHi-C data analysis, including cell embedding and cell type clustering on four independent scHi-C datasets. Taken together, scHiC-SSM represents a powerful tool to facilitate the study of 3D chromatin organization.
![在这里插入图片描述](https://img-blog.csdnimg.cn/7c298d757dd442a6ae7000a3a8b0bbcc.png#pic_center)

## 2. Environment setup
We recommend you to create virtual environments by [anaconda](https://docs.anaconda.com/anaconda/install/linux/). Also, make sure you have an NVIDIA GPU with Linux x86_64 Driver Version >= 470.103.01 (compatible with CUDA Version 11.3) if you want to accelarate training process.
#### 2.1 Install by conda

```powershell
git clone https://github.com/LyuHaoUZH/scHiC-SSM
conda env create -f scHiC-SSM_conda_environment.yml
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
#### 2.2 Install by pip
Alternatively, you can install the package by pip.

```powershell
git clone https://github.com/LyuHaoUZH/scHiC-SSM
pip install -r python-requirements.txt
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
## 3. Model training

```powershell
cd script

export INPUT_FILE=demo_data
export OUTPUT_FILE=test_result
export CELL_SUMMARY_FILE=supplementary_info/Li2019_Summary.txt
export GENOME_FILE=genome_info/hg19.chrom.sizes
export BAND=10
export CHROMOSOME="chr1"
export RESOLUTION=1000000
export LATENT=10
export CPUNUM=10

python scHiC-SSM.py \
    --inPath INPUT_FILE \
    --outdir OUTPUT_FILE \
    --cellSummary CELL_SUMMARY_FILE \
    --genome GENOME_FILE \
    --bandMax BAND \
    --chromList CHROMOSOME \
    --resolution RESOLUTION \
    --nLatent LATENT \
    --parallelCPU CPUNUM \
    --verbose
```
## 4. Data access
Users can acquire [data](https://drive.google.com/drive/folders/1fcq1gKC1OO89tFd3bEEtAWgnveJesSyG?usp=sharing%20Dependencies) or extend scHiC-SSM to their own datasets.
## 5. Citation
