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

# --- config 載入 ---
def load_config(config_path="config.json"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

# --- 使用者參數 ---
device_chan = config["device_chan"]  # NI myDAQ 通道
fs = config["fs"]  # 採樣率
record_len = config["record_len"]  # 記錄長度
chunk_size = config["chunk_size"]  # 每次讀取點數
buffer_size = config["buffer_size"]  # 硬體緩衝大小
mem_threshold_mb = config["mem_threshold_mb"]  # 記憶體上限
average_delay_time = config["average_delay_time"]  # 幾秒後開始計算平均
max_bpm = config["max_bpm"]  # 最大BPM上限
min_bpm = config["min_bpm"]  # 最小BPM上限
fft_freq_range = tuple(config["fft_freq_range"])  # fft顯示頻率範圍
fft_amp_range = tuple(config["fft_amp_range"])  # fft顯示震幅範圍
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
                    print("Read error, terminating read thread:", e)
                    break
                with buffer_lock:
                    data_buffer.extend(chunk)
            else:
                time.sleep(0.1)


# 啟動讀取執行緒
stop_event = threading.Event()
reader = threading.Thread(target=read_worker, args=(stop_event,), daemon=True)
reader.start()

# 設定畫布尺寸
fig = plt.figure(figsize=(14, 8))

# 使用 GridSpec 來設定子圖位置
gs = fig.add_gridspec(
    2, 2, height_ratios=[0.15, 0.85], width_ratios=[0.6, 0.4]
)  # 上半為按鈕區，下半為主圖和FFT圖

# BPM 顯示：左上 (gs[0,0])
ax_bpm = fig.add_subplot(gs[0, 0])
ax_bpm.axis("off")
bpm_display = ax_bpm.text(
    0.5, 0.5, "BPM: --", ha="center", va="center", fontsize=20, color="blue"
)

# 按鈕：右上 (gs[0,1])，FFT 正上方
ax_toggle = fig.add_subplot(gs[0, 1])
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
ax = fig.add_subplot(gs[1, 0])  # 主圖區
x_axis = np.linspace(-window_time, 0, record_len)
(line,) = ax.plot(x_axis, list(data_buffer), lw=1)
freq_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

# 設定主圖
ax.set_xlim(-window_time, 0)
ax.set_ylim(0, 5)
# ax.set_ylim(0.8, 1.2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
ax.set_title("Real-Time Waveform & Frequency Estimation")
ax.grid()

# 頻譜圖
ax_fft = fig.add_subplot(gs[1, 1])  # 右下
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
        print(f"[Warning] Memory usage {mem_mb:.1f} MB exceeds threshold, buffer cleared")
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

    # 頻率估計
    """
    detrended = raw_3s - np.mean(raw_3s)
    crossings = np.where((detrended[:-1] < 0) & (detrended[1:] >= 0))[0]
    if len(crossings) >= 2:
        diffs = np.diff(crossings)
        freqs = fs / diffs
        freq_est = np.mean(freqs)
        freq_std = np.std(freqs)

        if (
            running
            and (time.time_ns() - pause_time > average_delay_time * 1e9)
            and freq_est < max_bpm / 60.0
        ):
            freq_cum_sum += freq_est
            freq_cum_count += 1
        freq_avg = freq_cum_sum / freq_cum_count if freq_cum_count > 0 else 0.0

        freq_text.set_text(
            f"Freq: {freq_est:.4f} Hz ± {freq_std:.4f} Hz\nAvg: {freq_avg:.4f} Hz\nBPM: {freq_est*60:.2f}\nAvg_BPM: {freq_avg*60:.2f}"
        )
        bpm_display.set_text(f"BPM: {freq_est*60:.2f}")
    else:
        freq_text.set_text("Freq: -- Hz")
        bpm_display.set_text("BPM: --")"""

    # 計算FFT，僅使用前三秒的資料
    if running:  # 僅在開始時更新FFT
        N = len(raw_3s)
        signal_fft = np.fft.fft(raw_3s)
        freqs = np.fft.fftfreq(N, d=1 / fs)

        # 只取正頻率部分
        pos_freqs = freqs[: N // 2]
        pos_signal_fft = np.abs(signal_fft[: N // 2])

        # 限制顯示範圍，最大頻率為 max_f
        min_f = fft_freq_range[0]
        max_f = fft_freq_range[1]
        valid_indices = (pos_freqs >= min_f) & (pos_freqs <= max_f)
        pos_freqs = pos_freqs[valid_indices]
        pos_signal_fft = pos_signal_fft[valid_indices]

        # 畫出FFT頻譜
        ax_fft.clear()  # 清除上一輪的圖形
        ax_fft.plot(pos_freqs, pos_signal_fft)
        ax_fft.set_title("FFT Frequency Spectrum")
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Magnitude")
        ax_fft.grid(True)
        ax_fft.loglog()
        ax_fft.set_ylim(*fft_amp_range)
        ax_fft.grid(which="minor", axis="both", linestyle=":")
        # find main_freq
        max_index = pos_signal_fft.argmax()
        main_freq = pos_freqs[max_index]
        ax_fft.scatter([main_freq], [pos_signal_fft[max_index]], s=50)

        if (
            (time.time_ns() - pause_time > average_delay_time * 1e9)
            and main_freq < max_bpm / 60.0
            and main_freq > min_bpm / 60.0
        ):
            freq_cum_sum += main_freq
            freq_cum_count += 1
        freq_avg = freq_cum_sum / freq_cum_count if freq_cum_count > 0 else 0.0
        freq_text.set_text(
            f"Freq: {main_freq:.4f} Hz\nAvg: {freq_avg:.4f} Hz\nBPM: {main_freq*60:.2f}\nAvg_BPM: {freq_avg*60:.2f}"
        )
        bpm_display.set_text(f"BPM: {main_freq*60:.2f}")

    return line, freq_text, bpm_display


# 設置每50ms更新一次
ani = animation.FuncAnimation(fig, update, interval=10)
# , blit=True
try:
    plt.tight_layout()
    plt.show()
finally:
    stop_event.set()
    reader.join()

