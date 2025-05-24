# Real-Time NI myDAQ Data Acquisition & FFT Viewer

## 即時 NI myDAQ 資料擷取與 FFT 頻譜分析工具

本專案為使用 NI myDAQ 進行即時資料擷取、波形顯示與頻譜分析的 Python 工具，支援自訂參數與即時 BPM 顯示。  
This project is a Python tool that uses NI myDAQ for real-time data acquisition, waveform display, and frequency spectrum analysis, supporting customizable parameters and real-time BPM display.

> 本 README 文檔部分由 ChatGPT 生成  
> Part of this README was generated with the help of ChatGPT.

---

## 中文說明

### 1. 目錄結構

- `run.py`：主程式，負責資料擷取、繪圖、FFT 分析與互動操作。
- `config.json`：參數設定檔，可自訂硬體通道、取樣率、FFT 範圍等。
- `README.md`：專案說明文件。

### 2. 環境需求

- Python 3.9 以上
- NI-DAQmx 驅動與 Python 套件
- 主要套件：
  - numpy
  - matplotlib
  - nidaqmx
  - psutil

> 建議將 `nidaqmx` 套件直接安裝於 base（全域）環境，其餘可用虛擬環境安裝。

### 3. 安裝方式

```sh
pip install numpy matplotlib nidaqmx psutil
```

[參考影片](https://www.youtube.com/watch?v=hQMl3SHMLjg)

### 4. 參數設定

請編輯 [`config.json`](config.json) 檔案，調整硬體通道（`device_chan`）、取樣率（`fs`）、FFT 範圍等參數。

- `device_chan` 格式：  
  - 前半部為裝置名稱（請於 NI MAX 查詢，例如：`myDAQ1_youyouyou`）
  - 後半部為通道名稱（`/ai0` 或 `/ai1`）
  - 例如：`myDAQ1_youyouyou/ai0`

### 5. 執行方式

```sh
python run.py
```

執行後會開啟互動式視窗，按下 Start 開始擷取資料與繪圖。

### 6. 頻率計算方式簡介

本程式會先對原始訊號進行理想頻帶濾波（僅保留 `min_bpm` 與 `max_bpm` 對應的頻率範圍），再將濾波後的訊號進行去趨勢處理，最後利用過零點（zero-crossing）法計算主頻率。  
具體做法為：統計訊號在時間內通過零點的次數，計算平均週期，進而得到主頻率（即心跳頻率）。

---

## English documentation

### 1. Directory Structure

- `run.py`: Main script responsible for data acquisition, plotting, FFT analysis, and interactive control.
- `config.json`: Configuration file for hardware channels, sampling rate, FFT range, etc.
- `README.md`: Project documentation.

### 2. Environment Requirements

- Python 3.9 or above
- NI-DAQmx driver and Python package
- Main dependencies:
  - numpy
  - matplotlib
  - nidaqmx
  - psutil

> It is recommended to install the `nidaqmx` package directly in the base (global) environment. Other packages can be installed in a virtual environment.

### 3. Installation

```sh
pip install numpy matplotlib nidaqmx psutil
```

[reference video](https://www.youtube.com/watch?v=hQMl3SHMLjg)

### 4. Configuration Parameters

Edit the [`config.json`](config.json) file to customize hardware channel (`device_chan`), sampling rate (`fs`), FFT range, and other parameters.

- `device_chan` format:
  - The first part is the device name, which you can find in the NI MAX tool (e.g., `myDAQ1_youyouyou`).
  - The second part is the channel name. Use `/ai0` for analog input channel 0, `/ai1` for channel 1.
  - For example: `myDAQ1_youyouyou/ai0` indicates analog input channel 0 on device `myDAQ1_youyouyou`.

### 5. How to Run

```sh
python run.py
```

An interactive window will open. Click "Start" to begin data acquisition and plotting.

### 6. BPM Calculation Overview

The program first applies an ideal band-pass filter to the original signal (keeping only frequencies between `min_bpm` and `max_bpm`), then detrends the filtered signal, and finally estimates the dominant frequency using the zero-crossing method.  
Specifically, the algorithm counts the number of times the signal crosses zero within a time window to estimate the average period and hence determine the dominant frequency (i.e., heart rate).
