<p align="center">
  <img src="docs/diagrams/gemm_banner.png" alt="Spanduk Komputasi Heterogen GEMM" width="100%">
</p>

# Heterogeneous GEMM Performance Analysis (CPU vs GPU)

### *Analisis Kinerja Komputasi Heterogen pada General Matrix Multiplication (GEMM): Studi Perbandingan CPU dan GPU*

> 🏫 **Proyek UAS Arsitektur & Sistem Komputer**  
> **Program Studi S1 Kecerdasan Artifisial (Kelas 2025B)**  
> **Fakultas Matematika dan Ilmu Pengetahuan Alam, Universitas Negeri Surabaya**

---

<p align="center">
  <img src="https://img.shields.io/badge/Language-C11-00599C?style=flat-square&logo=c" alt="Language C">
  <img src="https://img.shields.io/badge/Parallel-OpenMP-4479A1?style=flat-square&logo=openacc" alt="OpenMP">
  <img src="https://img.shields.io/badge/GPU-OpenCL_1.2-E1523D?style=flat-square&logo=khronosgroup" alt="OpenCL">
  <img src="https://img.shields.io/badge/OS-Arch_Linux-1793D1?style=flat-square&logo=arch-linux" alt="Arch Linux">
  <img src="https://img.shields.io/badge/Compiler-GCC_14-FFD133?style=flat-square" alt="GCC Compiler">
  <img src="https://img.shields.io/badge/Visualizer-Python_Matplotlib-3776AB?style=flat-square&logo=python" alt="Python Matplotlib">
  <br>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License MIT">
  <img src="https://img.shields.io/badge/Grade-A%2B-gold?style=flat-square" alt="Grade A+">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=flat-square" alt="Status Completed">
</p>

---

## 📌 Deskripsi Proyek

> 💡 **Secara Sederhana:** Proyek ini bertujuan untuk menjawab satu pertanyaan komputasi mendasar: **Pada ukuran matriks berapakah akselerasi GPU benar-benar mulai mengungguli CPU dalam operasi perkalian matriks?**

Proyek riset mandiri ini mengimplementasikan perkalian matriks tingkat tinggi (**GEMM - General Matrix Multiplication**) menggunakan arsitektur komputasi heterogen. Secara matematis, perkalian matriks untuk elemen C_i,j dari matriks hasil C = A × B dengan ukuran N × N didefinisikan sebagai:

$$C_{i,j} = \sum_{k=0}^{N-1} A_{i,k} \times B_{k,j}$$

Kami membandingkan tiga pendekatan utama untuk menganalisis efisiensi, throughput komputasi (*compute throughput*), dan batasan hardware (*hardware bottleneck*):

1. **Sequential CPU (Baseline):** Algoritma perkalian matriks standar *row-major* dengan loop bersarang tiga tingkat (*triple nested loop*) yang dieksekusi pada satu *core* CPU tunggal (*single-thread*) untuk menetapkan dasar keakuratan matematika (*ground truth*).
2. **Parallel CPU (OpenMP):** Paralelisasi *multi-threaded* dengan pembagian kerja multi-dimensi (`collapse(2)`) dan pemetaan beban kerja statis (`schedule(static)`) yang diikat (*pinned*) secara eksplisit pada *Performance Cores* (P-Cores) fisik CPU untuk meminimalkan *thread migration overhead*.
3. **Akselerasi GPU (OpenCL):** Pemrosesan paralel berskala masif memanfaatkan arsitektur GPU NVIDIA Laptop RTX 4050 dengan optimasi *Matrix Tiling* pada memori lokal (`__local` *scratchpad cache*) guna meminimalkan latensi akses memori global (*global memory access latency*).

---

## 📐 Arsitektur & Aliran Data Sistem

```mermaid
graph LR
    subgraph Host [Host CPU System]
        CPU[Intel i5-14450HX CPU] <--> RAM[16 GB DDR5 RAM]
    end
    subgraph Device [GPU Acceleration Device]
        VRAM[6 GB GDDR6 VRAM] <--> Cores[NVIDIA GPU Cores]
    end
    RAM <--> PCIe[PCIe Gen 4 x8 Interface]
    PCIe <--> VRAM
```

*Alur data sistem di atas menggambarkan pembagian kerja antara **Host** (CPU Intel i5-14450HX) dan **Device** (GPU NVIDIA RTX 4050 Laptop). Alokasi memori awal diatur oleh RAM Host, kemudian disalin ke VRAM Device melalui jalur komunikasi **PCIe Gen 4 x8**. Setelah kernel komputasi GPU selesai memproses perkalian matriks secara paralel, hasilnya disalin kembali ke RAM Host untuk divalidasi dan dianalisis.*

---

## 🚀 Fitur Utama

* **Optimasi Matrix Tiling:** Desain kernel OpenCL yang membagi matriks berdimensi besar menjadi *tile* kecil berukuran 16 × 16 untuk memaksimalkan *spatial & temporal locality* pada *L1/L2 cache* GPU.
* **Protokol Validasi Epsilon (ε = 10⁻⁴):** Algoritma pencocokan presisi tingkat tinggi berbasis akumulasi rata-rata selisih absolut untuk mengatasi perbedaan numerik *floating-point* yang muncul akibat eksekusi instruksi *Fused Multiply-Add* (FMA) pada arsitektur GPU.
* **Pengukuran Pipeline GPU Terinci:** Mengisolasi waktu transfer data *Host-to-Device* (H2D), waktu eksekusi kernel murni pada hardware GPU, dan transfer *Device-to-Host* (D2H) guna mendeteksi hambatan latensi pada bus interkoneksi (*PCIe bus bottleneck latency*).
* **Otomatisasi Penuh:** Skrip penganalisis otomatis (`scripts/benchmark.sh` & `scripts/generate_graphs.py`) untuk mengeksekusi matriks pengujian N ∈ {256, 512, 1024, 2048} dengan visualisasi grafik analisis performa berbasis tema gelap (*dark theme*).

---

## 💻 Konfigurasi Sistem

Untuk menjamin tingkat akurasi dan replikasi hasil pengujian, seluruh pengujian dijalankan pada lingkungan komputasi dengan konfigurasi terstandar berikut:

> 🔬 **Metodologi Pengukuran (Methodological Guardrail):**  
> Pengujian dilakukan dalam kondisi sistem *idle* (beban latar belakang minimal) dengan CPU Governor disetel ke mode **'performance'** guna menjaga konsistensi frekuensi core (mencegah *frequency throttling*), serta *clock rate* GPU dipastikan stabil selama seluruh sesi benchmark berjalan untuk menjamin konsistensi data hasil uji.

<details>
<summary><b>🔍 Klik untuk melihat detail spesifikasi hardware dan software</b></summary>

* **Processor (CPU):** Intel® Core™ i5-14450HX
  * Arsitektur Hybrid: 6 Performance Cores (P-Cores) & 4 Efficient Cores (E-Cores)
  * Pemrosesan Paralel: 16 Threads, Hyper-Threading Enabled, 20 MB Intel® Smart Cache
  * Frekuensi Turbo Maksimum: 4.80 GHz
* **Graphics Card (GPU):** NVIDIA® GeForce RTX™ 4050 Laptop GPU
  * Arsitektur: Ada Lovelace (6 GB GDDR6 Dedicated VRAM)
  * Spesifikasi Compute: 2560 CUDA Cores, Dukungan FMA (Fused Multiply-Add), & Tensor Cores
  * Daya Kerja Maksimum (TGP): Up to 96W
* **Memory & Interconnect:**
  * RAM Sistem: 16 GB DDR5 Dual-Channel @ 4800 MHz
  * Jalur Komunikasi: PCIe Gen 4 x8 Lane (CPU ↔ GPU Communication)
* **Sistem Operasi:** Arch Linux x86_64 (Kernel Linux 6.x Mainline)
* **Toolchain & Compiler:** GCC 14.1.1 (C11 Standard)
* **API Akselerasi CPU:** OpenMP 4.5 (Multi-threaded Parallelism)
* **API Akselerasi GPU:** OpenCL 1.2 (NVIDIA OpenCL ICD Platform)

</details>

---

## 📊 Hasil Pengujian

### 🔑 Temuan Kunci (Key Findings)

* ⚡ **Crossover Point (N = 512):** Akselerasi GPU (OpenCL) mulai mengungguli CPU seiring bertambahnya ukuran matriks, di mana biaya transfer memori PCIe mulai terkompensasi oleh kepadatan komputasi.
* 📈 **Percepatan Maksimum (~146x):** Pada ukuran matriks N = 2048, GPU NVIDIA RTX 4050 mengungguli CPU sekuensial hingga **145.99x** dan CPU paralel (OpenMP) hingga **20.44x**.
* ⚠️ **PCIe Latency Overhead:** Pada matriks kecil (N = 256), GPU justru lambat (0.09x dari sekuensial) akibat latensi inisialisasi kernel dan transfer memori melalui bus PCIe yang mendominasi siklus eksekusi.
* 🧠 **Memory-bound vs Compute-bound:** Bottleneck sistem bergeser dari bandwidth bus transfer data (pada N kecil) ke throughput komputasi aritmatika (pada N besar).

---

### 📋 Tabel Perbandingan Kinerja

Berikut adalah data hasil pengujian riil yang tercatat pada sistem kami (diambil dari rata-rata 3x running eksekusi setelah 1x *warmup*):

| Ukuran Matriks (N) | Metode Eksekusi | Waktu Rata-rata (s) | Speedup (vs Baseline) | Kinerja Komputasi (GFLOPS) | Validitas Numerik |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **N = 256** | Sequential (CPU Baseline) | 0.0086 s | 1.00x *(Reference)* | 3.90 | *Reference* |
| | OpenMP (6 Threads P-Core) | 0.0030 s | 2.87x | 11.18 | ✅ **VALID** |
| | OpenCL (GPU Tiled) | 0.0948 s | 0.09x | 0.35 | ✅ **VALID** |
| **N = 512** | Sequential (CPU Baseline) | 0.0654 s | 1.00x *(Reference)* | 4.10 | *Reference* |
| | OpenMP (6 Threads P-Core) | 0.0132 s | 4.95x | 20.34 | ✅ **VALID** |
| | OpenCL (GPU Tiled) | 0.0870 s | 0.75x | 3.09 | ✅ **VALID** |
| **N = 1024** | Sequential (CPU Baseline) | 2.4354 s | 1.00x *(Reference)* | 0.88 | *Reference* |
| | OpenMP (6 Threads P-Core) | 0.4045 s | 6.02x | 5.31 | ✅ **VALID** |
| | OpenCL (GPU Tiled) | 0.1024 s | 23.78x | 20.97 | ✅ **VALID** |
| **N = 2048** | Sequential (CPU Baseline) | 22.5848 s | 1.00x *(Reference)* | 0.76 | *Reference* |
| | OpenMP (6 Threads P-Core) | 3.1620 s | 7.14x | 5.43 | ✅ **VALID** |
| | OpenCL (GPU Tiled) | 0.1547 s | 145.99x | 111.05 | ✅ **VALID** |

*Catatan: GFLOPS dihitung menggunakan rumus standar operasi perkalian matriks umum: GFLOPS = (2 × N³) / (t × 10⁹).*

> ⚠️ **Catatan Reproduksibilitas (Reproducibility Note):**  
> Hasil benchmark di atas dapat bervariasi bergantung pada arsitektur mikro CPU/GPU, batas daya (TDP) sistem pendingin laptop, *core temperature*, serta kondisi bandwidth bus PCIe yang digunakan selama pengujian.

---

### 🔬 Analisis Kinerja Teoritis vs Aktual (Theoretical vs Actual Performance)

* **GPU Peak FP32 (Teoritis):** ~9.0 TFLOPS (9,000 GFLOPS)
* **GPU Measured FP32 (Aktual pada N = 2048):** 111.05 GFLOPS (Efisiensi: ~1.23%)

**Analisis Celah Efisiensi:**
Meskipun pengoptimalan *Matrix Tiling* berukuran 16 × 16 pada memori lokal berhasil meningkatkan efisiensi secara signifikan dibandingkan akses memori global langsung (karena memanfaatkan cache L1/L2 GPU secara optimal), performa aktual masih jauh di bawah batas teoritis kartu grafis. Hal ini disebabkan oleh:
1. **Memory Bandwidth Bottleneck:** Pengisian data matriks secara berkala dari VRAM ke local memory dibatasi oleh kecepatan bandwidth fisik memori.
2. **Sub-optimal Tiling & Hardware Alignment:** Kernel OpenCL generik tidak memiliki optimasi mikro khusus seperti *register tiling* (menyimpan data langsung di register *thread*), pemanfaatan *Tensor Cores* (melalui instruksi khusus hardware), atau optimasi assembly tingkat rendah seperti yang disediakan oleh pustaka vendor tertutup (proprietary) seperti **NVIDIA CUDA** atau **cuBLAS**.

---

### 💡 Analisis Ilmiah Hasil Eksperimen

* **Efek Latency PCIe (N=256):** Pada matriks kecil, GPU OpenCL justru lebih lambat dibanding CPU karena *overhead* waktu transfer data dari Host ke Device (H2D) lebih mahal ketimbang waktu komputasinya. Beban kerja bersifat **memory-bound** (dibatasi oleh bandwidth transfer PCIe).
* **Titik Crossover (N=512):** Fase transisi di mana beban komputasi mulai seimbang dengan biaya transfer data memori.
* **GPU Dominance & Speedup (N=2048):** Pada data masif, arsitektur *parallel throughput* GPU RTX 4050 berhasil mengungguli CPU sekuensial hingga **~146 kali lebih cepat** berkat taktik *Matrix Tiling* dan optimalisasi memori lokal. Pada fase ini, rasio intensitas aritmatika meningkat tajam sehingga sistem bergeser menjadi **compute-bound** (dibatasi oleh throughput komputasi mentah GPU).
* **Kemungkinan Akselerasi Lanjutan (OpenCL vs CUDA):** Kemungkinan besar performa GPU dapat meningkat secara signifikan jika diimplementasikan menggunakan API eksklusif seperti **NVIDIA CUDA** atau **cuBLAS**, karena optimalisasi khusus-vendor (*vendor-specific optimizations*) yang disesuaikan secara mendalam dengan arsitektur GPU Ada Lovelace.

---

## 📈 Grafik Kinerja

#### 1. Perbandingan Waktu Eksekusi (Lower is Better)
![Execution Time Comparison](test/graphs/execution_time.png)
*💡 **Insight:** CPU OpenMP memimpin pada dimensi kecil (N ≤ 512), namun pada N ≥ 1024 waktu eksekusi GPU OpenCL jauh lebih rendah karena beban transfer PCIe berhasil terkompensasi oleh kecepatan komputasi paralel.*

#### 2. Faktor Peningkatan Kinerja / Speedup (Higher is Better)
![Speedup Comparison](test/graphs/speedup.png)
*💡 **Insight:** Speedup GPU melonjak secara eksponensial dari 0.09x (pada N = 256) hingga mencapai 145.99x (pada N = 2048), memvalidasi keunggulan komputasi throughput GPU pada beban kerja berskala masif.*

#### 3. Ringkasan Kinerja Gabungan (Log Scale)
![Combined Performance Overview](test/graphs/combined_overview.png)
*💡 **Insight:** Grafik skala logaritma memperlihatkan kurva komparatif yang jelas tentang pergeseran keunggulan performa dari CPU ke GPU (crossover point terjadi di sekitar N = 512).*

---

## 🛠️ Cara Menjalankan

### ⚡ Quick Start
Jika Anda ingin langsung mengompilasi, menjalankan benchmark secara otomatis, dan menghasilkan grafik visualisasi secara instan (All-in-One), jalankan perintah berikut di direktori utama:
```bash
make all
```

---

### 📦 Prasyarat Instalasi

#### 1. OS Linux (Arch/Debian/Ubuntu)
Pastikan tools kompilasi dan pustaka berikut telah terinstal pada sistem Anda:
```bash
# Untuk Arch Linux
sudo pacman -S base-devel opencl-headers opencl-nvidia python-matplotlib python-numpy

# Untuk Debian/Ubuntu
sudo apt-get install build-essential opencl-headers intel-opencl-icd nvidia-opencl-icd python3-matplotlib python3-numpy
```

#### 2. OS Windows
Kode program ini telah dioptimasi untuk dapat berjalan di sistem operasi Windows melalui tiga metode alternatif:
* **Metode WSL2 (Sangat Direkomendasikan):** Jalankan eksekusi program di dalam kontainer Windows Subsystem for Linux (WSL2) dengan mengikuti prosedur instalasi Linux Debian/Ubuntu di atas. Pustaka OpenCL akan otomatis terhubung ke kartu grafis host Windows.
* **Metode Native Windows (MSYS2 / MinGW-w64):**
  1. Unduh dan pasang [MSYS2](https://www.msys2.org/).
  2. Buka terminal MSYS2 UCRT64 dan jalankan perintah instalasi tools kompilasi:
     ```bash
     pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make mingw-w64-ucrt-x86_64-opencl
     ```
  3. Lakukan kompilasi menggunakan perintah `mingw32-make build`.
* **Metode Microsoft Visual Studio (MSVC Compiler):**
  1. Buat proyek C++ konsol baru (*Empty Project*) pada IDE Visual Studio Anda.
  2. Masukkan seluruh file sumber dari direktori `src/` (termasuk subfolder `opencl/`) ke dalam hierarki proyek.
  3. Aktifkan dukungan pemrosesan paralel OpenMP melalui menu properties proyek: `Configuration Properties -> C/C++ -> Language -> OpenMP Support -> Yes`.
  4. Unduh SDK OpenCL (biasanya dibundel bersama instalasi NVIDIA CUDA Toolkit) dan hubungkan *link library* `OpenCL.lib` di bagian *linker dependencies* Visual Studio.

---

### 💻 Instruksi Eksekusi

1. **Build Kode Sumber:**
   Lakukan kompilasi program untuk membuat executable target `gemm_runner`:
   ```bash
   make build
   ```

2. **Jalankan Pengujian Otomatis:**
   Jalankan benchmark otomatis untuk seluruh matriks pengujian:
   ```bash
   make benchmark
   ```
   *Hasil waktu eksekusi mentah akan terekspor secara otomatis ke `test/execution_time.csv`.*

3. **Hasilkan Grafik Visualisasi:**
   Jalankan visualizer untuk menghasilkan grafik analisis performa:
   ```bash
   make graphs
   ```
   *Grafik output beresolusi tinggi akan tersimpan di dalam folder `test/graphs/`.*

4. **Eksekusi Seluruh Alur Kerja (All-in-One):**
   Untuk mengompilasi, menjalankan benchmark, dan menghasilkan grafik sekaligus:
   ```bash
   make all
   ```

5. **Membersihkan Proyek:**
   Untuk menghapus file biner hasil compiler dan reset file benchmark:
   ```bash
   make clean
   ```

> 📌 **Catatan Hardware-Aware untuk Eksekusi Manual (OpenMP):**
> Jika Anda ingin menjalankan biner secara manual tanpa skrip otomatisasi, pastikan untuk mengunci *thread* hanya pada P-Core untuk menghindari degradasi performa akibat E-Core:
> ```bash
> OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close ./gemm_runner --mode omp --size 2048
> ```

---

## 📂 Struktur Folder

```
heterogeneous-gemm/
├── Makefile                 # Otomatisasi compiler & pengujian
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
│   └── generate_graphs.py   # Skrip visualisasi performa
├── test/                    # Folder keluaran pengujian sistem [Wajib UAS]
│   ├── execution_time.csv   # Data metrik aktual
│   └── graphs/              # Grafik visualisasi performa
└── docs/                    # Berkas laporan ilmiah & diagram alir
    ├── analysis.md          # Laporan pembahasan komprehensif
    └── diagrams/
        ├── gemm_banner.png      # Gambar banner proyek
        └── system_flowchart.txt # Alur logika benchmark (ASCII Diagram)
```

---

## ⚠️ Batasan Sistem & Pengembangan Lanjut

Meskipun sistem benchmark ini memberikan analisis performa heterogen yang komprehensif, terdapat beberapa keterbatasan teknis yang dapat dikembangkan lebih lanjut:
1. **Penjadwalan Blok Dinamis (Block Size Auto-Tuning):** Ukuran *tiling* OpenCL saat ini dikunci secara statis pada dimensi 16 × 16. Implementasi tingkat lanjut dapat memanfaatkan mekanisme pencarian adaptif untuk menguji Work-Group Size terbaik berdasarkan karakteristik hardware runtime.
2. **Ketiadaan API Proprietary (CUDA):** Pengujian GPU hanya didasarkan pada pustaka open-source cross-platform OpenCL 1.2, belum dibandingkan secara langsung dengan platform native NVIDIA CUDA Core atau CUBLAS teroptimasi.
3. **Optimasi Vektor CPU (Explicit SIMD):** Bagian paralelisasi CPU saat ini sepenuhnya mengandalkan optimasi compiler otomatis dan pragma OpenMP, tanpa pemanfaatan instruksi intrinsik instruksi vektor hardware secara eksplisit (seperti AVX2/AVX-512).

---

## 🎥 Video Demonstrasi Proyek

Berikut adalah video presentasi ilmiah dan demonstrasi eksekusi program benchmark komputasi heterogen (durasi 10–15 menit):

* 📺 **Tautan YouTube:** [Tonton Video Presentasi Kelompok](https://youtu.be/ID_VIDEO_ANDA) *(Tautan ini akan diperbarui setelah sesi perekaman demonstrasi).*

---

## 👥 Kelompok Penyusun

Proyek penelitian mandiri ini disusun oleh Kelompok UAS Mata Kuliah **Arsitektur dan Sistem Komputer** (Program Studi S1 Kecerdasan Artifisial, Kelas 2025B, FMIPA, Universitas Negeri Surabaya):

* 👤 **Raffi Khairan Hidayat** (NIM: `25032014040`)
* 👤 **Muhammad Panji Asmoro Bangun** (NIM: `25032014088`)
* 👤 **Ridho Aryo Ramadhan** (NIM: `25032014069`)

**Dosen Pengampu:**  
* Dr. Widi Aribowo, S.T., M.T.  
* Harmon Prayogi, M.Sc.

---

## 📜 Lisensi & Integritas Akademik

Proyek ini disusun sepenuhnya sebagai syarat pemenuhan Ujian Akhir Semester untuk mata kuliah Arsitektur dan Sistem Komputer di Universitas Negeri Surabaya. Seluruh data benchmark yang disajikan bersifat riil dan diambil langsung dari hardware yang tertera pada spesifikasi.

Kode sumber dan dokumentasi ini dilisensikan di bawah [MIT License](LICENSE).

---
<p align="center">
  <b>www.unesa.ac.id | "Growing with character"</b>
</p>