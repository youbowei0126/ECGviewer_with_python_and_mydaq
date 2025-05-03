# Real-Time NI myDAQ Data Acquisition & FFT Viewer

本專案為使用 NI myDAQ 進行即時資料擷取、波形顯示與頻譜分析的 Python 工具，支援自訂參數與即時 BPM 顯示。

本文檔部分生成自chatgpt

## 1. 目錄結構

- `run.py`：主程式，負責資料擷取、繪圖、FFT 分析與互動操作。
- `config.json`：參數設定檔，可自訂硬體通道、取樣率、FFT 範圍等。
- `README.md`：專案說明文件。


## 2. 環境需求

- Python 3.9 以上
- NI-DAQmx 驅動與 Python 套件
- 主要套件：
  - numpy
  - matplotlib
  - nidaqmx
  - psutil

---
### 2.設定mydaq的環境設定

關於mydaq的環境設定教學請看[這裡](https://www.youtube.com/watch?v=hQMl3SHMLjg)

https://www.youtube.com/watch?v=hQMl3SHMLjg
---
### 2.2安裝套件

```sh
pip install numpy matplotlib nidaqmx psutil
```


> 注意：建議使用虛擬環境安裝 `numpy`、`matplotlib`、`psutil` 等套件。但 `nidaqmx` 建議直接安裝於 base（全域）環境


## 3.設定參數

請透過編輯 [`config.json`](config.json) 檔案，調整硬體通道（`device_chan`）、取樣率（`fs`）、FFT 範圍等相關參數。

- `device_chan` 格式說明：  
  - 前半部為裝置名稱，請於 NI MAX 工具中查詢您的 myDAQ 裝置名稱（例如：`myDAQ1`）。
  - 後半部為通道名稱，類比輸入通道 0 請填寫 `/ai0`，通道 1 請填寫 `/ai1`。
  - 例如：`myDAQ1/ai0` 表示 myDAQ1 裝置的類比輸入通道 0。




## 4.執行方式

```sh
python run.py
```

執行後會開啟互動式視窗，按下 Start 開始擷取資料與繪圖。
