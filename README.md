# Alpha-XIC

Alpha-XIC is a deep neural network to score the coelution of peak groups which improves the identification of DIA data by DIA-NN.

## Hardware

A [GPU with CUDA support](https://developer.nvidia.com/cuda-gpus)

## Package

- [PyTorch](https://pytorch.org/get-started/locally/#windows-anaconda) 1.0.0+
- [Pyteomics](https://pyteomics.readthedocs.io/en/latest/)
- [numba](http://numba.pydata.org/)

## Tutorial

1. Compile the modified DIA-NN: 
    ```shell script
    cd 'Alpha-XIC/diann_1.7.16_plus_alpha/mstoolkit'
    make
    ```
2. Make a workspace folder containing:
    - diann-alpha-xic.exe compiled by step 1
    - *.mzML, 
    - lib.tsv (the spectral library)
    
3. Run DIA-NN:
    ```shell script
   cd workspace
    ./diann-alpha-xic.exe --f *.mzML --lib lib.tsv --out diann_out.tsv --threads 4 --qvalue 0.01
    ```
   Meanwhile, the modified DIA-NN will generate the scores file in workspace.

4. Run Alpha-xic:
    ```shell script
    cd 'Alpha-XIC' project
   python main_alpha workspace_dir
    ```
