# Alpha-XIC

Alpha-XIC is a deep neural network to score the coelution of peak groups which improves the identification of DIA data.

## Hardware

A [GPU with CUDA support](https://developer.nvidia.com/cuda-gpus)

## Package

- [PyTorch](https://pytorch.org/get-started/locally/#windows-anaconda) 1.0.0+
- [Pyteomics](https://pyteomics.readthedocs.io/en/latest/)
- [Cython](https://cython.org/)

## Tutorial

1. Make a workspace folder containing: 
    - *.mzML, 
    - osw.tsv (the output by OpenSWATH v2.4.0)
    - lib.tsv (the spectral library)
2. Compile the modified PyProphet:
    ```shell script
    cd 'Alpha-XIC/pyprophet'
    python setup.py build_ext --inplace
    ```
3. Run:
    ```shell script
    cd 'Alpha-XIC'
    python main.py your_workspace_dir
    ```