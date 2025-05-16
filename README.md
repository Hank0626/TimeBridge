# TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting

<div align="center">

**[<a href="https://mp.weixin.qq.com/s/bCEWRvU-dBNwa2FxwaTMHQ">中文解读1</a>]**
**[<a href="https://mp.weixin.qq.com/s/oFw5rXvbtqgL8clhucsAnQ">中文解读2</a>]**
**[<a href="https://www.bilibili.com/video/BV12GC6YuEiB/?spm_id_from=333.337.search-card.all.click&vd_source=42dea39777f3aa2191db3d7e7e283b66">BiliBIli Video</a>]**
</div>

## Updates

🚩 **2025-05-01:** TimeBridge has been accepted as **ICML 2025 Poster**. 

🚩 **2025-04-18:** Release the detailed training logs (see [_logs](./_logs/)).

🚩 **2025-02-11:** Release the code.

🚩 **2024-10-08:** Initial upload to arXiv [[PDF]](https://arxiv.org/abs/2410.04442).

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
@article{liu2025timebridge,
      title={TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting}, 
      author={Liu, Peiyuan and Wu, Beiliang and Hu, Yifan and Li, Naiqi and Dai, Tao and Bao, Jigang and Xia, Shu-Tao},
      journal={International Conference on Machine Learning},
      year={2025},
}
```

## Contact
If you have any questions, please get in touch with [lpy23@mails.tsinghua.edu.cn](lpy23@mails.tsinghua.edu.cn) or submit an issue.
