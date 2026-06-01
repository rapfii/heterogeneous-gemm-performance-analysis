# Analisis Kinerja Komputasi Heterogeneous GEMM (CPU vs GPU)

Ujian Akhir Semester Genap Tahun Akademik 2025/2026 — **Arsitektur dan Sistem Komputer**  
**Program Studi S1 Kecerdasan Artifisial (Kelas 2025B)**, Fakultas Matematika dan Ilmu Pengetahuan Alam, **Universitas Negeri Surabaya**

<p align="center">
  <img src="docs/diagrams/gemm_banner.png" alt="GEMM Heterogeneous Banner" width="100%">
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/Language-C11-00599C?style=flat-square&logo=c" alt="Language C">
  <img src="https://img.shields.io/badge/Parallel-OpenMP-4479A1?style=flat-square&logo=openacc" alt="OpenMP">
  <img src="https://img.shields.io/badge/GPU-OpenCL_1.2-E1523D?style=flat-square&logo=khronosgroup" alt="OpenCL">
  <img src="https://img.shields.io/badge/OS-Arch_Linux-1793D1?style=flat-square&logo=arch-linux" alt="Arch Linux">
  <img src="https://img.shields.io/badge/Compiler-GCC_14-FFD133?style=flat-square" alt="GCC Compiler">
  <img src="https://img.shields.io/badge/Visualizer-Python_Matplotlib-3776AB?style=flat-square&logo=python" alt="Python Matplotlib">
</p>

---

## 👥 Kelompok Penyusun

* **Raffi Khairan Hidayat** — NIM: `25032014040`
* **Muhammad Panji Asmoro Bangun** — NIM: `25032014088`
* **Ridho Aryo Ramadhan** — NIM: `25032014069`

**Program Studi S1 Kecerdasan Artifisial (Kelas 2025B)**  
Fakultas Matematika dan Ilmu Pengetahuan Alam, **Universitas Negeri Surabaya**  
**Dosen Pengampu:** Dr. Widi Aribowo, S.T., M.T. & Harmon Prayogi, M.Sc.  
*Slogan Universitas: "Growing with character"*  

---

## 🎥 Video Demonstrasi Proyek

> [!IMPORTANT]
> **Tautan Video YouTube (Unlisted/Public):**  
> Tempelkan link video presentasi kelompok Anda di sini setelah melakukan rekaman (durasi 10–15 menit).  
> format: `https://youtu.be/<ID_VIDEO_ANDA>`

---

## 📌 Deskripsi Proyek

Proyek riset mandiri ini mengimplementasikan perkalian matriks tingkat tinggi (**GEMM - General Matrix Multiplication**) menggunakan arsitektur komputasi heterogen. Kami membandingkan tiga pendekatan utama untuk menganalisis efisiensi, throughput komputasi, dan batasan hardware (bottleneck):

1. **Sequential CPU (Baseline):** Algoritma perkalian matriks standar row-major tiga tingkat loop tunggal yang berjalan pada satu core CPU untuk menetapkan dasar keakuratan matematika (Ground Truth).
2. **Parallel CPU (OpenMP):** Paralelisasi multi-threaded dengan pembagian kerja dinamis (`collapse(2)` dan `schedule(static)`) pada core CPU fisik terdedikasi.
3. **GPU Accelerated (OpenCL):** Pemrosesan paralel masif menggunakan arsitektur GPU NVIDIA Laptop RTX 4050 dengan optimasi **Matrix Tiling** memanfaatkan local memory (`__local` scratchpad cache) untuk mengurangi global memory access latency.

---

## 🚀 Fitur Utama Sistem

* **Matriks Tiling Teroptimasi:** Desain kernel OpenCL yang membagi matriks besar menjadi tile kecil berukuran $16 \times 16$ untuk memaksimalkan *spatial & temporal locality* pada L1/L2 cache GPU.
* **Protokol Validasi Epsilon ($\varepsilon = 10^{-4}$):** Algoritma pencocokan presisi tingkat tinggi berbasis akumulasi rata-rata selisih absolut untuk mengatasi perbedaan numerik floating-point yang muncul akibat instruksi Fused Multiply-Add (FMA) pada GPU vs Standard CPU.
* **Pengukuran Pipeline GPU Terinci:** Mengisolasi waktu transfer data Host-to-Device (H2D), waktu eksekusi kernel murni pada hardware GPU, dan transfer Device-to-Host (D2H) guna mendeteksi latency bottleneck PCIe bus.
* **Otomatisasi Penuh:** Skrip penganalisis otomatis (`scripts/benchmark.sh` & `scripts/generate_graphs.py`) untuk mengeksekusi matriks pengujian $N \in \{256, 512, 1024, 2048\}$ dengan visualisasi grafik bernuansa **dark clean elegant**.

---

## 💻 Spesifikasi Sistem Pengujian

* **CPU:** Intel® Core™ i5-14450HX (Hybrid Architecture: Performance & Efficiency Cores, Hyper-Threading Aktif)
* **GPU:** NVIDIA® GeForce RTX™ 4050 Laptop GPU (6GB GDDR6, Arsitektur Ada Lovelace)
* **OS / Compiler:** Arch Linux / GCC 14.1.1
* **API Akselerasi:** OpenCL 1.2 CUDA / OpenMP 4.5

---

## 📊 Hasil Pengujian Sistem (Rata-rata 3x Running)

Berikut adalah data hasil pengujian riil yang tercatat pada sistem kami:

| Metode Eksekusi | N = 256 | N = 512 | N = 1024 | N = 2048 | Validitas Numerik |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sequential (CPU Baseline)** | 0.0086 s | 0.0654 s | 2.4354 s | 22.5848 s | *Reference* |
| **OpenMP (6 Threads P-Core)** | 0.0030 s | 0.0132 s | 0.4045 s | 3.1620 s | ✅ **VALID** |
| **OpenCL (GPU Tiled)** | 0.0948 s | 0.0870 s | 0.1024 s | 0.1547 s | ✅ **VALID** |

### 📈 Grafik Kinerja (Dark Clean Elegant Theme)

#### 1. Perbandingan Waktu Eksekusi (Lower is Better)
![Execution Time Comparison](test/graphs/execution_time.png)

#### 2. Faktor Peningkatan Kinerja / Speedup (Higher is Better)
![Speedup Comparison](test/graphs/speedup.png)

#### 3. Ringkasan Kinerja Gabungan (Log Scale)
![Combined Performance Overview](test/graphs/combined_overview.png)

---

## 🛠️ Langkah-Langkah Menjalankan Sistem

### 📦 Prasyarat Instalasi
Untuk sistem Linux berbasis Arch/Debian, pastikan pustaka berikut terinstal:
```bash
# Untuk Arch Linux
sudo pacman -S base-devel opencl-headers opencl-nvidia python-matplotlib python-numpy

# Untuk Debian/Ubuntu
sudo apt-get install build-essential opencl-headers intel-opencl-icd nvidia-opencl-icd python3-matplotlib python3-numpy
```

### 1. Build Kode Sumber
Lakukan kompilasi program untuk membuat executable target `gemm_runner`:
```bash
make build
```

### 2. Jalankan Pengujian Otomatis
Jalankan benchmark otomatis untuk seluruh matriks pengujian:
```bash
make benchmark
```
*Hasil waktu eksekusi mentah akan terekspor secara otomatis ke `test/execution_time.csv`.*

### 3. Hasilkan Grafik Visualisasi
Jalankan visualizer untuk melahirkan grafik performa bernuansa gelap:
```bash
make graphs
```
*Grafik output beresolusi tinggi akan tersimpan di dalam folder `test/graphs/`.*

### 4. Eksekusi Seluruh Alur Kerja (All-in-One)
Untuk mengompilasi, menjalankan benchmark, dan menghasilkan grafik sekaligus:
```bash
make all
```

### 5. Membersihkan Proyek
Untuk menghapus file biner hasil kompilasi dan reset file benchmark:
```bash
make clean
```

---

## 📂 Struktur Folder Proyek

```
heterogeneous-gemm/
├── Makefile                 # Otomatisasi kompilasi & pengujian
├── README.md                # Dokumentasi utama proyek UAS
├── src/                     # Seluruh kode sumber C & OpenCL kernel
│   ├── main.c               # Driver benchmark utama
│   ├── gemm_common.h        # Alokasi memori & validasi epsilon
│   ├── sequential.c         # Fungsi perkalian sequential CPU
│   ├── openmp.c             # Paralelisasi multi-core OpenMP
│   └── opencl/
│       ├── opencl_host.c    # Pipeline Host GPU OpenCL
│       └── gemm_kernel.cl   # Kernel GPU Tiling
├── scripts/                 # Otomatisasi penganalisis data
│   ├── benchmark.sh         # Pengumpul metrik dalam bentuk CSV
│   └── generate_graphs.py   # Visualizer grafik berkelas
├── test/                    # Folder keluaran pengujian sistem [Wajib UAS]
│   ├── execution_time.csv   # Data metrik aktual
│   └── graphs/              # Hasil plot dark elegant
└── docs/                    # Berkas laporan ilmiah & diagram alir
    ├── analysis.md          # Laporan pembahasan komprehensif
    └── diagrams/
        └── system_flowchart.txt # Alur logika benchmark (ASCII Diagram)
```

---

> www.unesa.ac.id | **"Growing with character"**
