# Real-Time NI myDAQ Data Acquisition & FFT Viewer

本專案為使用 NI myDAQ 進行即時資料擷取、波形顯示與頻譜分析的 Python 工具，支援自訂參數與即時 BPM 顯示。

本README文檔部分生成自chatgpt

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
### 2.1 設定mydaq的環境設定

關於mydaq的環境設定教學請看[這裡](https://www.youtube.com/watch?v=hQMl3SHMLjg)

https://www.youtube.com/watch?v=hQMl3SHMLjg

`nidaqmx` 建議直接安裝於 base（全域）環境

---
### 2.2 安裝套件

```sh
pip install numpy matplotlib nidaqmx psutil
```


> 注意：可使用虛擬環境安裝 `numpy`、`matplotlib`、`psutil` 等套件。但 `nidaqmx` 建議直接安裝於 base（全域）環境


## 3. 設定參數

請透過編輯 [`config.json`](config.json) 檔案，調整硬體通道（`device_chan`）、取樣率（`fs`）、FFT 範圍等相關參數。

- `device_chan` 格式說明：  
  - 前半部為裝置名稱，請於 NI MAX 工具中查詢您的 myDAQ 裝置名稱（例如：`myDAQ1_youyouyou`）。
  - 後半部為通道名稱，類比輸入通道 0 請填寫 `/ai0`，通道 1 請填寫 `/ai1`。
  - 例如：`myDAQ1_youyouyou/ai0` 表示 myDAQ1_youyouyou 裝置的類比輸入通道 0。




## 4. 執行方式

```sh
python run.py
```

執行後會開啟互動式視窗，按下 Start 開始擷取資料與繪圖。

## 頻率計算方式簡介

本程式會先對原始訊號進行理想頻帶濾波（僅保留 `min_bpm` 與 `max_bpm` 對應的頻率範圍），再將濾波後的訊號進行去趨勢處理，最後利用過零點（zero-crossing）法計算主頻率。  
具體做法為：統計訊號在時間內通過零點的次數，計算平均週期，進而得到主頻率（即心跳頻率）。
