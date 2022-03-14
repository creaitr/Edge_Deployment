# PyTorch

Environments setting

## conda setting
- Command lines \
    Create an env and 
    ```
    conda create -n [name] python=[3.8]
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    ```

- Use yml file \
    [environmet.yml](environment.yml)
    ```
    name: jb
    channels:
        - pytorch
        - defaults
    dependencies:
        #...
        - cudatoolkit=10.2.89=hfd86e86_1
        #...
        - pip=21.0.1=py38h06a4308_0
        - python=3.8.8=hdb3f193_5
        - pytorch=1.8.1=py3.8_cuda10.2_cudnn7.6.5_0
        #...
        - torchaudio=0.8.1=py38
        - torchvision=0.9.1=py38_cu102
    ```
    
    Create an env
    ```
    ```