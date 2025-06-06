import threading
import time
from collections import deque
import json
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
import psutil
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import subprocess
import scipy.signal  # 請確保已安裝 scipy


# --- config 載入 ---
def load_config(config_path="config.json"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到設定檔: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config_data, config_path="config.json"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)


config = load_config()

# --- 使用者參數 ---
device_chan = config["device_chan"]  # NI myDAQ 通道
fs = config["fs"]  # 採樣率
record_len = config["record_len"]  # 記錄長度
chunk_size = config["chunk_size"]  # 每次讀取點數
buffer_size = config["buffer_size"]  # 硬體緩衝大小
mem_threshold_mb = config["mem_threshold_mb"]  # 記憶體上限
average_delay_time = config["average_delay_time"]  # 幾秒後開始計算平均
frequence_upper = config["frequence_upper"]  # 頻率上限
frequence_lower = config["frequence_lower"]  # 頻率下限

max_bpm = frequence_upper*60  # 最大BPM上限
min_bpm = frequence_lower*60  # 最小BPM上限
fft_freq_range = tuple(config["fft_freq_range"])  # fft顯示頻率範圍
fft_amp_range = tuple(config["fft_amp_range"])  # fft顯示震幅範圍
show_filtered_data = config["show_filtered_data"]  # 是否顯示濾波後的資料
filtered_data_baseline = config["filtered_data_baseline"]  # 濾波過後的頻率要顯示的位移
show_filtered_fft=config["show_filtered_fft"]
filter_type = config.get("filter_type", "band_pass")  # 預設為帶通濾波器
remove_dc= config.get("remove_dc", True)  # 是否移除直流成分（零頻率）
filter_model = config.get("filter_model", "ideal")  # 濾波器模型
butter_order = config.get("butter_order", 4)  # Butterworth 階數
fir_numtaps = config.get("fir_numtaps", 51)  # FIR 濾波器 tap 數（刪除 ma_n）

# 計算窗口時間
window_time = record_len / fs
initial_error = 1.0 / window_time

# 緩衝區
data_buffer = deque([0.0] * record_len, maxlen=record_len)
buffer_lock = threading.Lock()

# 狀態變數
running = False
freq_cum_sum = 0.0
freq_cum_count = 0
pause_time = time.time_ns()


# 背景讀取執行緒
def read_worker(stop_event):
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(device_chan)
        task.timing.cfg_samp_clk_timing(
            rate=fs, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=buffer_size
        )
        task.in_stream.input_buf_size = buffer_size

        while not stop_event.is_set():
            if running:
                try:
                    chunk = task.read(
                        number_of_samples_per_channel=chunk_size, timeout=1.0
                    )
                except Exception as e:
                    print("讀取錯誤，終止讀取執行緒：", e)
                    break
                with buffer_lock:
                    data_buffer.extend(chunk)
            else:
                time.sleep(0.1)


# 濾波器函數
def apply_filter(data, filter_type, filter_model, fs, freq_lower, freq_upper, butter_order=4, fir_numtaps=51, remove_dc=True):
    """
    應用不同類型和模型的濾波器
    
    Parameters:
    - data: 輸入信號
    - filter_type: 濾波器類型 ("low_pass", "high_pass", "band_pass", "band_stop")
    - filter_model: 濾波器模型 ("ideal", "butterworth", "fir")
    - fs: 採樣頻率
    - freq_lower: 下限頻率
    - freq_upper: 上限頻率
    - butter_order: Butterworth 濾波器階數
    - fir_numtaps: FIR 濾波器 tap 數
    - remove_dc: 是否移除直流成分
    """
    
    if filter_model == "ideal":
        # 理想濾波器 (FFT-based)
        N = len(data)
        fft_data = np.fft.fft(data)
        freqs = np.fft.fftfreq(N, d=1/fs)
        
        # 根據濾波器類型建立遮罩
        if filter_type == "low_pass":
            mask = np.abs(freqs) <= freq_upper
        elif filter_type == "high_pass":
            mask = np.abs(freqs) >= freq_lower
        elif filter_type == "band_pass":
            mask = (np.abs(freqs) >= freq_lower) & (np.abs(freqs) <= freq_upper)
        elif filter_type == "band_stop":
            mask = (np.abs(freqs) <= freq_lower) | (np.abs(freqs) >= freq_upper)
        else:
            mask = np.ones_like(freqs, dtype=bool)
        
        if remove_dc:
            mask = mask & (freqs != 0)
        
        fft_data_filtered = fft_data * mask
        filtered = np.fft.ifft(fft_data_filtered).real
        
    elif filter_model == "butterworth":
        # Butterworth 濾波器
        nyquist = fs / 2
        
        if filter_type == "low_pass":
            if freq_upper >= nyquist:
                freq_upper = nyquist * 0.99
            b, a = scipy.signal.butter(butter_order, freq_upper/nyquist, btype='low')
        elif filter_type == "high_pass":
            if freq_lower <= 0:
                freq_lower = 0.01
            b, a = scipy.signal.butter(butter_order, freq_lower/nyquist, btype='high')
        elif filter_type == "band_pass":
            if freq_lower <= 0:
                freq_lower = 0.01
            if freq_upper >= nyquist:
                freq_upper = nyquist * 0.99
            b, a = scipy.signal.butter(butter_order, [freq_lower/nyquist, freq_upper/nyquist], btype='band')
        elif filter_type == "band_stop":
            if freq_lower <= 0:
                freq_lower = 0.01
            if freq_upper >= nyquist:
                freq_upper = nyquist * 0.99
            b, a = scipy.signal.butter(butter_order, [freq_lower/nyquist, freq_upper/nyquist], btype='bandstop')
        else:
            # 如果類型不正確，返回原始數據
            filtered = data.copy()
            
        if 'b' in locals() and 'a' in locals():
            filtered = scipy.signal.filtfilt(b, a, data)
        else:
            filtered = data.copy()
            
        if remove_dc:
            filtered = filtered - np.mean(filtered)
            
    elif filter_model == "fir":
        # FIR 濾波器 (使用窗函數法)
        try:
            if filter_type == "low_pass":
                # 低通濾波器
                coeffs = scipy.signal.firwin(fir_numtaps, cutoff=freq_upper, fs=fs, pass_zero='lowpass')
            elif filter_type == "high_pass":
                # 高通濾波器
                coeffs = scipy.signal.firwin(fir_numtaps, cutoff=freq_lower, fs=fs, pass_zero='highpass')
            elif filter_type == "band_pass":
                # 帶通濾波器
                if freq_lower >= freq_upper:
                    print(f"警告：freq_lower ({freq_lower}) >= freq_upper ({freq_upper})，使用預設值")
                    freq_lower = freq_upper * 0.1
                coeffs = scipy.signal.firwin(fir_numtaps, cutoff=[freq_lower, freq_upper], fs=fs, pass_zero='bandpass')
            elif filter_type == "band_stop":
                # 帶阻濾波器
                if freq_lower >= freq_upper:
                    print(f"警告：freq_lower ({freq_lower}) >= freq_upper ({freq_upper})，使用預設值")
                    freq_lower = freq_upper * 0.1
                coeffs = scipy.signal.firwin(fir_numtaps, cutoff=[freq_lower, freq_upper], fs=fs, pass_zero='bandstop')
            else:
                print(f"警告：未知的濾波器類型 {filter_type}，返回原始數據")
                filtered = data.copy()
                return filtered
            
            # 應用 FIR 濾波器
            filtered = scipy.signal.lfilter(coeffs, 1.0, data)
            
        except Exception as e:
            print(f"FIR 濾波器錯誤：{e}，改用理想濾波器")
            return apply_filter(data, filter_type, "ideal", fs, freq_lower, freq_upper, butter_order, fir_numtaps, remove_dc)
        
        if remove_dc:
            filtered = filtered - np.mean(filtered)
    else:
        # 未知模型，返回原始數據
        print(f"警告：未知的濾波器模型 {filter_model}，返回原始數據")
        filtered = data.copy()
    
    return filtered


# 設定視窗 (修改)
def show_settings():
    global device_chan, fs, record_len, chunk_size, buffer_size, mem_threshold_mb
    global average_delay_time, max_bpm, min_bpm, fft_freq_range, fft_amp_range
    global show_filtered_data, filtered_data_baseline, window_time
    global filter_type, filter_model, butter_order, remove_dc, fir_numtaps
    
    # 暫停數據擷取
    was_running = running
    if running:
        toggle(None)  # 暫停擷取
    
    # 建立設定視窗
    root = tk.Tk()
    root.title("Setting")
    root.geometry("1600x1200")
    
    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # 建立捲動區域
    canvas = tk.Canvas(main_frame)
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # 建立設定項目
    entries = {}
    
    row = 0
    ttk.Label(scrollable_frame, text="Device Settings", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 5))
    row += 1

    ttk.Label(scrollable_frame, text="Device Channel:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
    entries["device_chan"] = ttk.Entry(scrollable_frame, width=30)
    entries["device_chan"].insert(0, device_chan)
    entries["device_chan"].grid(row=row, column=1, sticky="w", padx=5, pady=5)
    row += 1

    ttk.Label(scrollable_frame, text="Sampling Rate (Hz):").grid(row=row, column=0, sticky="w", padx=5, pady=5)
    entries["fs"] = ttk.Entry(scrollable_frame, width=15)
    entries["fs"].insert(0, str(fs))
    entries["fs"].grid(row=row, column=1, sticky="w", padx=5, pady=5)
    row += 1

    ttk.Label(scrollable_frame, text="Record Length (points):").grid(row=row, column=0, sticky="w", padx=5, pady=5)
    entries["record_len"] = ttk.Entry(scrollable_frame, width=15)
    entries["record_len"].insert(0, str(record_len))
    entries["record_len"].grid(row=row, column=1, sticky="w", padx=5, pady=5)
    row += 1

    ttk.Label(scrollable_frame, text="Chunk Size:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
    entries["chunk_size"] = ttk.Entry(scrollable_frame, width=15)
    entries["chunk_size"].insert(0, str(chunk_size))
    entries["chunk_size"].grid(row=row, column=1, sticky="w", padx=5, pady=5)
    row += 1

    ttk.Label(scrollable_frame, text="Hardware Buffer Size:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
    entries["buffer_size"] = ttk.Entry(scrollable_frame, width=15)
    entries["buffer_size"].insert(0, str(buffer_size))
    entries["buffer_size"].grid(row=row, column=1, sticky="w", padx=5, pady=5)
    row += 1

    ttk.Label(scrollable_frame, text="Memory Limit (MB):").grid(row=row, column=0, sticky="w", padx=5, pady=5)
    entries["mem_threshold_mb"] = ttk.Entry(scrollable_frame, width=15)
    entries["mem_threshold_mb"].insert(0, str(mem_threshold_mb))
    entries["mem_threshold_mb"].grid(row=row, column=1, sticky="w", padx=5, pady=5)
    row += 1

    ttk.Label(scrollable_frame, text="BPM Settings", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 5))
    row += 1

    ttk.Label(scrollable_frame, text="Average Delay Time (s):").grid(row=row, column=0, sticky="w", padx=5, pady=5)
    entries["average_delay_time"] = ttk.Entry(scrollable_frame, width=15)
    entries["average_delay_time"].insert(0, str(average_delay_time))
    entries["average_delay_time"].grid(row=row, column=1, sticky="w", padx=5, pady=5)
    row += 1

    ttk.Label(scrollable_frame, text="Filter Settings", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 5))
    row += 1

    # Filter Type
    ttk.Label(scrollable_frame, text="Filter Type:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
    filter_options = ["low_pass", "high_pass", "band_pass", "band_stop"]
    filter_type_var = tk.StringVar(value=filter_type)
    entries["filter_type"] = ttk.Combobox(scrollable_frame, values=filter_options, textvariable=filter_type_var, width=15)
    entries["filter_type"].grid(row=row, column=1, sticky="w", padx=5, pady=5)
    row += 1

    # Filter Model - 適用於所有濾波器類型
    ttk.Label(scrollable_frame, text="Filter Model:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
    filter_model_options = ["ideal", "butterworth", "fir"]
    filter_model_var = tk.StringVar(value=filter_model)
    entries["filter_model"] = ttk.Combobox(scrollable_frame, values=filter_model_options, textvariable=filter_model_var, width=15)
    entries["filter_model"].grid(row=row, column=1, sticky="w", padx=5, pady=5)
    row += 1

    # 新增 frequence_upper (Hz)
    ttk.Label(scrollable_frame, text="Frequency Upper (Hz):").grid(row=row, column=0, sticky="w", padx=5, pady=5)
    entries["frequence_upper"] = ttk.Entry(scrollable_frame, width=15)
    entries["frequence_upper"].insert(0, str(frequence_upper))
    entries["frequence_upper"].grid(row=row, column=1, sticky="w", padx=5, pady=5)
    row += 1

    # 新增 frequence_lower (Hz)
    ttk.Label(scrollable_frame, text="Frequency Lower (Hz):").grid(row=row, column=0, sticky="w", padx=5, pady=5)
    entries["frequence_lower"] = ttk.Entry(scrollable_frame, width=15)
    entries["frequence_lower"].insert(0, str(frequence_lower))
    entries["frequence_lower"].grid(row=row, column=1, sticky="w", padx=5, pady=5)
    row += 1

    # Butterworth order
    ttk.Label(scrollable_frame, text="Butterworth Order:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
    butter_order_var = tk.IntVar(value=butter_order)
    entries["butter_order"] = ttk.Entry(scrollable_frame, width=15, textvariable=butter_order_var)
    entries["butter_order"].grid(row=row, column=1, sticky="w", padx=5, pady=5)
    row += 1

    # Moving average n
    ttk.Label(scrollable_frame, text="Fir Numtaps\n(should be odd):").grid(row=row, column=0, sticky="w", padx=5, pady=5)
    fir_numtaps_var = tk.IntVar(value=fir_numtaps)
    entries["fir_numtaps"] = ttk.Entry(scrollable_frame, width=15, textvariable=fir_numtaps_var)
    entries["fir_numtaps"].grid(row=row, column=1, sticky="w", padx=5, pady=5)
    row += 1

    # --- 將 Display Settings 移到第二欄 ---
    display_col = 2
    display_row = 0
    ttk.Label(scrollable_frame, text="FFT Display Settings", font=("Arial", 12, "bold")).grid(row=display_row, column=display_col, columnspan=2, sticky="w", pady=(10, 5))
    display_row += 1

    ttk.Label(scrollable_frame, text="Frequency Range Min:").grid(row=display_row, column=display_col, sticky="w", padx=5, pady=5)
    entries["fft_freq_min"] = ttk.Entry(scrollable_frame, width=15)
    entries["fft_freq_min"].insert(0, str(fft_freq_range[0]))
    entries["fft_freq_min"].grid(row=display_row, column=display_col + 1, sticky="w", padx=5, pady=5)
    display_row += 1

    ttk.Label(scrollable_frame, text="Frequency Range Max:").grid(row=display_row, column=display_col, sticky="w", padx=5, pady=5)
    entries["fft_freq_max"] = ttk.Entry(scrollable_frame, width=15)
    entries["fft_freq_max"].insert(0, str(fft_freq_range[1]))
    entries["fft_freq_max"].grid(row=display_row, column=display_col + 1, sticky="w", padx=5, pady=5)
    display_row += 1

    ttk.Label(scrollable_frame, text="Amplitude Range Min:").grid(row=display_row, column=display_col, sticky="w", padx=5, pady=5)
    entries["fft_amp_min"] = ttk.Entry(scrollable_frame, width=15)
    entries["fft_amp_min"].insert(0, str(fft_amp_range[0]))
    entries["fft_amp_min"].grid(row=display_row, column=display_col + 1, sticky="w", padx=5, pady=5)
    display_row += 1

    ttk.Label(scrollable_frame, text="Amplitude Range Max:").grid(row=display_row, column=display_col, sticky="w", padx=5, pady=5)
    entries["fft_amp_max"] = ttk.Entry(scrollable_frame, width=15)
    entries["fft_amp_max"].insert(0, str(fft_amp_range[1]))
    entries["fft_amp_max"].grid(row=display_row, column=display_col + 1, sticky="w", padx=5, pady=5)
    display_row += 1

    show_filtered_fft_var = tk.BooleanVar(value=show_filtered_fft)
    ttk.Label(scrollable_frame, text="Show Filtered FFT in Spectrum:").grid(row=display_row, column=display_col, sticky="w", padx=5, pady=5)
    entries["show_filtered_fft"] = ttk.Checkbutton(scrollable_frame, variable=show_filtered_fft_var)
    entries["show_filtered_fft"].grid(row=display_row, column=display_col + 1, sticky="w", padx=5, pady=5)
    display_row += 1

    ttk.Label(scrollable_frame, text="Waveform Display Settings", font=("Arial", 12, "bold")).grid(row=display_row, column=display_col, columnspan=2, sticky="w", pady=(10, 5))
    display_row += 1
    
    show_filter_var = tk.BooleanVar(value=show_filtered_data)
    ttk.Label(scrollable_frame, text="Show Filtered Data:").grid(row=display_row, column=display_col, sticky="w", padx=5, pady=5)
    entries["show_filtered_data"] = ttk.Checkbutton(scrollable_frame, variable=show_filter_var)
    entries["show_filtered_data"].grid(row=display_row, column=display_col + 1, sticky="w", padx=5, pady=5)
    display_row += 1

    ttk.Label(scrollable_frame, text="Filtered Data Baseline:").grid(row=display_row, column=display_col, sticky="w", padx=5, pady=5)
    entries["filtered_data_baseline"] = ttk.Entry(scrollable_frame, width=15)
    entries["filtered_data_baseline"].insert(0, str(filtered_data_baseline))
    entries["filtered_data_baseline"].grid(row=display_row, column=display_col + 1, sticky="w", padx=5, pady=5)
    display_row += 1

    remove_dc_var = tk.BooleanVar(value=remove_dc)
    ttk.Label(scrollable_frame, text="Remove DC (zero frequency):").grid(row=display_row, column=display_col, sticky="w", padx=5, pady=5)
    entries["remove_dc"] = ttk.Checkbutton(scrollable_frame, variable=remove_dc_var)
    entries["remove_dc"].grid(row=display_row, column=display_col + 1, sticky="w", padx=5, pady=5)
    display_row += 1

    button_frame = ttk.Frame(root)
    button_frame.pack(fill=tk.X, padx=10, pady=10)
    
    def save_settings():
        try:
            new_config = {
                "device_chan": entries["device_chan"].get(),
                "fs": float(entries["fs"].get()),
                "record_len": int(entries["record_len"].get()),
                "chunk_size": int(entries["chunk_size"].get()),
                "buffer_size": int(entries["buffer_size"].get()),
                "mem_threshold_mb": int(entries["mem_threshold_mb"].get()),
                "average_delay_time": int(entries["average_delay_time"].get()),
                "frequence_upper": float(entries["frequence_upper"].get()),
                "frequence_lower": float(entries["frequence_lower"].get()),
                "fft_freq_range": [float(entries["fft_freq_min"].get()), float(entries["fft_freq_max"].get())],
                "fft_amp_range": [float(entries["fft_amp_min"].get()), float(entries["fft_amp_max"].get())],
                "show_filtered_data": show_filter_var.get(),
                "filtered_data_baseline": float(entries["filtered_data_baseline"].get()),
                "show_filtered_fft": show_filtered_fft_var.get(),
                "filter_type": entries["filter_type"].get(),
                "remove_dc": remove_dc_var.get(),
                "filter_model": entries["filter_model"].get(),
                "butter_order": int(entries["butter_order"].get()),
                "fir_numtaps": int(entries["fir_numtaps"].get()),  # 改為 fir_numtaps
            }

            # 儲存設定到 config.json
            save_config(new_config)

            # 更新全域變數
            global device_chan, fs, record_len, chunk_size, buffer_size, mem_threshold_mb
            global average_delay_time, frequence_upper, frequence_lower, fft_freq_range, fft_amp_range
            global show_filtered_data, filtered_data_baseline, window_time
            global filter_type, show_filtered_fft, remove_dc
            global filter_model, butter_order, fir_numtaps  # 改為 fir_numtaps

            device_chan = new_config["device_chan"]
            fs = new_config["fs"]
            record_len = new_config["record_len"]
            chunk_size = new_config["chunk_size"]
            buffer_size = new_config["buffer_size"]
            mem_threshold_mb = new_config["mem_threshold_mb"]
            average_delay_time = new_config["average_delay_time"]
            frequence_upper = new_config["frequence_upper"]
            frequence_lower = new_config["frequence_lower"]
            fft_freq_range = tuple(new_config["fft_freq_range"])
            fft_amp_range = tuple(new_config["fft_amp_range"])
            show_filtered_data = new_config["show_filtered_data"]
            filtered_data_baseline = new_config["filtered_data_baseline"]
            show_filtered_fft = new_config["show_filtered_fft"]
            filter_type = new_config["filter_type"]
            remove_dc = new_config["remove_dc"]
            filter_model = new_config["filter_model"]
            butter_order = new_config["butter_order"]
            fir_numtaps = new_config["fir_numtaps"]  # 改為 fir_numtaps
            window_time = record_len / fs

            if messagebox.askyesno("Success", "Settings saved! Do you want to restart the program now?"):
                root.destroy()
                plt.close('all')
                stop_event.set()
                reader.join()
                python = sys.executable
                os.execl(python, python, *sys.argv)
            else:
                root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Error occurred while saving settings: {str(e)}")
    
    ttk.Button(button_frame, text="Save", command=save_settings).pack(side=tk.RIGHT, padx=5)
    ttk.Button(button_frame, text="Cancel", command=root.destroy).pack(side=tk.RIGHT, padx=5)

    root.mainloop()


# 啟動讀取執行緒
stop_event = threading.Event()
reader = threading.Thread(target=read_worker, args=(stop_event,), daemon=True)
reader.start()

# 設定畫布尺寸
fig = plt.figure(figsize=(14, 8))

# 使用 GridSpec 來設定子圖位置
gs = fig.add_gridspec(
    2, 3, height_ratios=[0.15, 0.85], width_ratios=[0.4, 0.2, 0.4]
)  # 上半為按鈕區，下半為主圖和FFT圖

# BPM 顯示：左上 (gs[0,0])
ax_bpm = fig.add_subplot(gs[0, 0])
ax_bpm.axis("off")
bpm_display = ax_bpm.text(
    0.5, 0.5, "BPM: --", ha="center", va="center", fontsize=20, color="blue"
)

# 設定按鈕：中上 (gs[0,1])
ax_settings = fig.add_subplot(gs[0, 1])
ax_settings.set_xticks([])
ax_settings.set_yticks([])
for spine in ax_settings.spines.values():
    spine.set_visible(True)

# 建立設定按鈕
btn_settings = Button(
    ax_settings,
    "Settings",
    color="lightgray",
    hovercolor="0.975",
)
btn_settings.on_clicked(lambda event: show_settings())

# 控制按鈕：右上 (gs[0,2])，FFT 正上方
ax_toggle = fig.add_subplot(gs[0, 2])
# 隱掉刻度，但保留邊框
ax_toggle.set_xticks([])
ax_toggle.set_yticks([])
for spine in ax_toggle.spines.values():
    spine.set_visible(True)

# 建立一個有背景色和 hover 效果的 Button
btn_toggle = Button(
    ax_toggle,
    "Start",
    color="lightgray",  # 按鈕背景色
    hovercolor="0.975",  # 滑鼠移上去的顏色
)

# 主圖
ax = fig.add_subplot(gs[1, 0:2])  # 主圖區
x_axis = np.linspace(-window_time, 0, record_len)
(line,) = ax.plot(x_axis, list(data_buffer), lw=1)
freq_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

# 設定主圖
ax.set_xlim(-window_time, 0)
ax.set_ylim(0.5, 2.5)
ax.set
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
ax.set_title("Real-Time Waveform & Frequency Estimation")
ax.grid()

# 頻譜圖
ax_fft = fig.add_subplot(gs[1, 2])  # 右下
ax_fft.set_title("FFT Frequency Spectrum")
ax_fft.set_xlabel("Frequency (Hz)")
ax_fft.set_ylabel("Magnitude")
ax_fft.grid(True)

# 微調畫布，確保子圖之間有適當的間距
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)


# 按鈕回調
def toggle(event):
    global running, freq_cum_sum, freq_cum_count, pause_time
    running = not running
    print("Acquisition running:", running)
    if running:
        with buffer_lock:
            data_buffer.clear()
            data_buffer.extend([0.0] * record_len)
        freq_cum_sum = 0.0
        freq_cum_count = 0
        pause_time = time.time_ns()
        btn_toggle.label.set_text("Pause")
    else:
        btn_toggle.label.set_text("Start")
    fig.canvas.draw()


# 設定按鈕回調
btn_toggle.on_clicked(toggle)


# 更新動畫
def update(frame):
    global freq_cum_sum, freq_cum_count, pause_time
    # 記憶體檢查
    mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    if mem_mb > mem_threshold_mb:
        with buffer_lock:
            data_buffer.clear()
            data_buffer.extend([0.0] * record_len)
        print(f"[警告] 記憶體 {mem_mb:.1f} MB 超過阈值，已清空緩衝區")
        freq_cum_sum, freq_cum_count = 0, 0
        return line, freq_text, bpm_display

    # 取得前三秒的數據
    with buffer_lock:
        raw = np.array(data_buffer)

    # 截取前三秒的資料
    num_samples = int(3 * fs)  # 3秒的資料
    if len(raw) >= num_samples:
        raw_3s = raw[-num_samples:]  # 取最後3秒的數據
    else:
        raw_3s = raw  # 如果資料不足3秒，直接使用現有資料

    line.set_ydata(raw)

    # 使用新的濾波器函數
    min_freq = min_bpm / 60.0
    max_freq = max_bpm / 60.0
    
    # 應用濾波器
    filtered = apply_filter(
        raw_3s, 
        filter_type, 
        filter_model, 
        fs, 
        frequence_lower, 
        frequence_upper, 
        butter_order, 
        fir_numtaps,  # 改為 fir_numtaps
        remove_dc
    )


    # 計算過零點頻率
    detrended = filtered - np.mean(filtered)
    crossings = np.where((detrended[:-1] < 0) & (detrended[1:] >= 0))[0]

    # 利用 crossings 計算頻率
    if len(crossings) > 1:
        period = (
            (crossings[-1] - crossings[0]) / (len(crossings) - 1) / fs
        )  # 平均週期（秒）
        zero_cross_freq = 1.0 / period if period > 0 else 0.0
    else:
        zero_cross_freq = 0.0
        
    # 計算FFT，僅使用前三秒的資料
    if running:  # 僅在開始時更新FFT
        N = len(raw_3s)
        signal_fft = np.fft.fft(raw_3s)
        freqs = np.fft.fftfreq(N, d=1 / fs)

        # 只取正頻率部分
        pos_freqs = freqs[: N // 2]
        pos_signal_fft = np.abs(signal_fft[: N // 2])

        # 新增：計算濾波後的頻譜
        filtered_fft = np.fft.fft(filtered)
        pos_filtered_fft = np.abs(filtered_fft[: N // 2])

        # 限制顯示範圍，最大頻率為 max_f
        min_f = fft_freq_range[0]
        max_f = fft_freq_range[1]
        valid_indices = (pos_freqs >= min_f) & (pos_freqs <= max_f)
        pos_freqs = pos_freqs[valid_indices]
        pos_signal_fft = pos_signal_fft[valid_indices]
        pos_filtered_fft = pos_filtered_fft[valid_indices]

        # 畫出FFT頻譜
        ax_fft.clear()  # 清除上一輪的圖形

        # 新增：在 min_bpm~max_bpm 對應的頻率區間加上半透明底色
        min_freq_band = min_bpm / 60.0
        max_freq_band = max_bpm / 60.0
        if filter_type == "band_pass":
        
            ax_fft.axvspan(
                min_freq_band, max_freq_band, color="orange", alpha=0.3, label="BPM Range"
            )
        elif filter_type == "band_stop":
            ax_fft.axvspan(
                0, min_freq_band, color="orange", alpha=0.3, label="BPM Range"
            )
            ax_fft.axvspan(
                max_freq_band, max_f, color="orange", alpha=0.3, label="BPM Range"
            )
            
        elif filter_type == "low_pass":
            ax_fft.axvspan(
                0, max_freq_band, color="orange", alpha=0.3, label="BPM Range"
            )
        elif filter_type == "high_pass":
            ax_fft.axvspan(
                min_freq_band, max_f, color="orange", alpha=0.3, label="BPM Range"
            )
        else:
            ax_fft.axvspan(
                0, max_f, color="orange", alpha=0.3, label="BPM Range"
            )
            
            
        ax_fft.plot(pos_freqs, pos_signal_fft, label="Raw FFT")
        if show_filtered_fft:
            ax_fft.plot(pos_freqs, pos_filtered_fft, label="Filtered FFT", color="red", linestyle="--", lw=3)
        ax_fft.set_title("FFT Frequency Spectrum")
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Magnitude")
        ax_fft.grid(True)
        ax_fft.loglog()
        # ax_fft.set_ylim(*fft_amp_range)
        ax_fft.set_ylim(1e-5,1e5)
        ax_fft.grid(which="minor", axis="both", linestyle=":")
        ax_fft.legend()
        
        # find main_freq
        if len(pos_signal_fft) > 0:
            max_index = pos_signal_fft.argmax()
            main_freq = pos_freqs[max_index]
            ax_fft.scatter([main_freq], [pos_signal_fft[max_index]], s=50)

        if (
            (time.time_ns() - pause_time > average_delay_time * 1e9)
            and zero_cross_freq < max_bpm / 60.0
            and zero_cross_freq > min_bpm / 60.0
        ):
            freq_cum_sum += zero_cross_freq
            freq_cum_count += 1
        freq_avg = freq_cum_sum / freq_cum_count if freq_cum_count > 0 else 0.0
        freq_text.set_text(
            f"Freq: {zero_cross_freq:.4f} Hz\nAvg: {freq_avg:.4f} Hz\nBPM: {zero_cross_freq*60:.2f}\nAvg_BPM: {freq_avg*60:.2f}"
        )
        bpm_display.set_text(f"BPM: {zero_cross_freq*60:.2f}")

    # 新增：畫出filtered資料
    if show_filtered_data:
        if not hasattr(update, "filtered_line"):
            # 第一次呼叫時建立filtered的線，並上移filtered_data_baseline
            (update.filtered_line,) = ax.plot(
                x_axis[-len(filtered) :],
                filtered + filtered_data_baseline,
                lw=1,
                color="orange",
                label="Filtered",
            )
            # 畫出filtered base line（filtered_data_baseline）並用虛線
            (update.filtered_baseline,) = ax.plot(
                x_axis[-len(filtered) :],
                np.full_like(filtered, filtered_data_baseline),
                lw=1,
                color="orange",
                linestyle="--",
                label="Filtered base",
            )
            ax.legend(loc="upper right")
        else:
            update.filtered_line.set_ydata(
                np.pad(
                    filtered + filtered_data_baseline,
                    (len(raw) - len(filtered), 0),
                    "constant",
                    constant_values=np.nan,
                )
            )
            update.filtered_baseline.set_ydata(
                np.pad(
                    np.full_like(filtered, filtered_data_baseline),
                    (len(raw) - len(filtered), 0),
                    "constant",
                    constant_values=np.nan,
                )
            )

    return line, freq_text, bpm_display


# 設置每10ms更新一次
ani = animation.FuncAnimation(fig, update, interval=10)

try:
    plt.tight_layout()
    plt.show()
finally:
    stop_event.set()
    reader.join()
