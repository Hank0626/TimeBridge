# TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting
[[中文解读1](https://mp.weixin.qq.com/s/bCEWRvU-dBNwa2FxwaTMHQ)][[中文解读2](https://mp.weixin.qq.com/s/oFw5rXvbtqgL8clhucsAnQ)]

## Updates/News 🆕

🚩 **Updates** (2025-02-11) Release the code.

🚩 **Updates** (2024-10-08) Initial upload to arXiv [[PDF]](https://arxiv.org/abs/2410.04442).

## Usage

1. Install the dependencies
    ```bash
    pip install -r requirements.txt
    ```

2. Obtain the dataset from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) and extract it to the root directory of the project. Make sure the extracted folder is named `dataset` and has the following structure:
    ```
    dataset
    ├── electricity
    │   └── electricity.csv
    ├── ETT-small
    │   ├── ETTh1.csv
    │   ├── ETTh2.csv
    │   ├── ETTm1.csv
    │   └── ETTm2.csv
    ├── PEMS
    │   ├── PEMS03.npz
    │   ├── PEMS04.npz
    │   ├── PEMS07.npz
    │   └── PEMS08.csv
    ├── Solar
    │   └── solar_AL.txt
    ├── traffic
    │   └── traffic.csv
    └── weather
        └── weather.csv
    ```

3. Train and evaluate the model. All the training scripts are located in the `scripts` directory. For example, to train the model on the Solar-Energy dataset, run the following command:
    ```bash
    sh ./scripts/TimeBridge.sh
    ```


## Bibtex
If you find this work useful, please consider citing it:

```
@article{liu2024time,
      title={TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting}, 
      author={Liu, Peiyuan and Wu, Beiliang and Hu, Yifan and Li, Naiqi and Dai, Tao and Bao, Jigang and Xia, Shu-Tao},
      journal={arXiv preprint arXiv:2410.04442},
      year={2024},
      arxiv={2410.04442}
}
```

## Contact
If you have any questions, please contact [lpy23@mails.tsinghua.edu.cn](lpy23@mails.tsinghua.edu.cn) or submit an issue.


