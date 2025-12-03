# Real-time rPPG Heart Rate Detection

Sistem deteksi detak jantung real-time menggunakan webcam dengan metode rPPG (remote Photoplethysmography).

## Penjelasan Program

Program ini melakukan deteksi heart rate secara real-time menggunakan webcam dengan metode **POS (Plane-Orthogonal-to-Skin)**. Metode POS memproyeksikan sinyal RGB ke plane yang orthogonal terhadap warna kulit untuk mengurangi motion artifacts dan perubahan iluminasi.

**Perbedaan program ini dengan yang dilakukan di kelas:** Program ini menggunakan metode POS yang memproses kombinasi RGB channels, input real-time dari webcam, full facial ROI dengan skin segmentation, visualisasi real-time berupa plots (signal & spectrum) dan overlay UI, serta keyboard controls untuk interaksi dengan sistem.

## Alur Sistem

1. **Face Detection**: Mendeteksi wajah menggunakan MediaPipe Face Mesh dan mengekstrak ROI (Region of Interest)
2. **Skin Segmentation**: Memfilter piksel non-kulit menggunakan HSV color range untuk meningkatkan akurasi
3. **Spatial Averaging**: Menghitung rata-rata nilai RGB dari piksel ROI yang telah difilter
4. **Signal Normalization**: Menormalisasi sinyal RGB untuk menghilangkan DC component
5. **POS Method**: Menerapkan transformasi POS untuk mengekstrak sinyal pulse
6. **Detrending**: Menghilangkan trend signal menggunakan moving average
7. **Bandpass Filtering**: Memfilter sinyal dengan Butterworth bandpass filter (0.67-4.0 Hz / 40-240 BPM)
8. **FFT Analysis**: Menganalisis spektrum frekuensi untuk mendeteksi dominant frequency
9. **Heart Rate Estimation**: Mengkonversi peak frequency menjadi BPM dan smoothing dengan moving average

## Visualisasi

### Plot Real-time
- **Upper Plot**: Filtered Pulse Signal (sinyal pulse setelah filtering)
- **Lower Plot**: Power Spectrum (spektrum frekuensi dengan peak detection)

### Video Overlay
- ROI mask (overlay hijau transparan pada area wajah)
- Heart rate (BPM)
- Buffer size progress
- FPS counter
- Status sistem

## Keyboard Controls

| Key | Fungsi |
|-----|--------|
| `q` | Quit aplikasi |
| `s` | Save frame saat ini |
| `r` | Reset sistem |
| `p` | Toggle visualization on/off |

## Installation

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aloisiusedwin/Multimedia.git
   cd Multimedia/rppg
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   For a more efficient setup, we recommend using the `uv` virtual environment tool:
   ```bash
   uv pip install -r requirements.txt
   ```
   Don't have `uv` installed? Follow the [Installation | uv](https://docs.astral.sh/uv/getting-started/installation/)

3. **Run the Application**:
   ```bash
   python main.py
   ```
