<p align="center">
  <img src="docs/diagrams/gemm_banner.png" alt="Spanduk Komputasi Heterogen GEMM" width="100%">
</p>

# Heterogeneous GEMM Performance Analysis (CPU vs GPU)

### *Analisis Kinerja Komputasi Heterogen pada General Matrix Multiplication (GEMM): CPU & GPU Comparative Study*

> 🏫 **Proyek UAS Arsitektur & Sistem Komputer**  
> **Program Studi S1 Kecerdasan Artifisial (Kelas 2025B)**  
> **Fakultas Matematika dan Ilmu Pengetahuan Alam, Universitas Negeri Surabaya**

---

<p align="center">
  <img src="https://img.shields.io/badge/Language-C11-00599C?style=flat-square&logo=c" alt="Language C">
  <img src="https://img.shields.io/badge/Parallel-OpenMP-4479A1?style=flat-square&logo=openacc" alt="OpenMP">
  <img src="https://img.shields.io/badge/GPU-OpenCL_3.0-E1523D?style=flat-square&logo=khronosgroup" alt="OpenCL">
  <img src="https://img.shields.io/badge/OS-Arch_Linux-1793D1?style=flat-square&logo=arch-linux" alt="Arch Linux">
  <img src="https://img.shields.io/badge/Compiler-GCC_16-FFD133?style=flat-square" alt="GCC Compiler">
  <img src="https://img.shields.io/badge/Visualizer-Python_Matplotlib-3776AB?style=flat-square&logo=python" alt="Python Matplotlib">
  <br>
</p>

---

## 🎥 Video Demonstrasi Proyek

Berikut adalah video presentasi ilmiah dan demonstrasi running program benchmark komputasi heterogen:

* 📺 **Tautan YouTube:** [Tonton Video Presentasi Kelompok](https://youtu.be/DGlnM8gVhfk)
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

## 📌 Deskripsi Proyek

> 💡 **Secara Sederhana:** Proyek ini bertujuan untuk menjawab satu pertanyaan komputasi mendasar: **Pada ukuran matriks berapakah akselerasi GPU benar-benar mulai mengungguli CPU dalam operasi perkalian matriks?**

Proyek riset mandiri ini mengimplementasikan perkalian matriks tingkat tinggi (**GEMM - General Matrix Multiplication**) menggunakan arsitektur komputasi heterogen. Secara matematis, perkalian matriks untuk elemen C_i,j dari matriks hasil C = A × B dengan ukuran N × N didefinisikan sebagai:

$$C_{i,j} = \sum_{k=0}^{N-1} A_{i,k} \times B_{k,j}$$

Kami membandingkan tiga pendekatan utama untuk menganalisis efisiensi, throughput komputasi (*compute throughput*), dan batasan hardware (*hardware bottleneck*):

1. **Sequential CPU (Baseline):** Algoritma perkalian matriks standar *row-major* dengan loop bersarang tiga tingkat (*triple nested loop*) yang dijalankan pada satu *core* CPU tunggal (*single-thread*) untuk menetapkan dasar keakuratan matematika (*ground truth*).
2. **Parallel CPU (OpenMP):** Paralelisasi *multi-threaded* dengan pembagian kerja multi-dimensi (`collapse(2)`) and static workload mapping (`schedule(static)`) yang diikat (*pinned*) secara eksplisit pada *Performance Cores* (P-Cores) fisik CPU untuk meminimalkan *thread migration overhead*.
3. **Akselerasi GPU (OpenCL):** Parallel processing berskala masif memanfaatkan arsitektur GPU NVIDIA Laptop RTX 4050 dengan optimasi *Matrix Tiling* pada memori lokal (`__local` *scratchpad cache*) guna meminimalkan latensi akses memori global (*global memory access latency*).

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

*System data flow di atas menggambarkan workload distribution antara **Host** (CPU Intel i5-14450HX) dan **Device** (GPU NVIDIA RTX 4050 Laptop). Memory allocation awal diatur oleh RAM Host, kemudian disalin ke VRAM Device melalui interconnect **PCIe Gen 4 x8**. Setelah kernel komputasi GPU selesai memproses perkalian matriks secara paralel, hasilnya disalin kembali ke RAM Host untuk divalidasi dan dianalisis.*

---

## 🚀 Fitur Utama

* **Optimasi Matrix Tiling:** Desain kernel OpenCL yang membagi matriks berdimensi besar menjadi *tile* kecil berukuran 16 × 16 untuk memaksimalkan *spatial & temporal locality* pada *L1/L2 cache* GPU.
* **Protokol Validasi Epsilon (ε = 10⁻⁴):** Algoritma pencocokan presisi tingkat tinggi berbasis akumulasi rata-rata selisih absolut untuk mengatasi perbedaan numerik *floating-point* yang muncul akibat running instruksi *Fused Multiply-Add* (FMA) pada arsitektur GPU.
* **Pengukuran Pipeline GPU Terinci:** Mengisolasi waktu transfer data *Host-to-Device* (H2D), waktu running kernel murni pada hardware GPU, dan transfer *Device-to-Host* (D2H) guna mendeteksi hambatan latensi pada bus interkoneksi (*PCIe bus bottleneck latency*).
* **Otomatisasi Penuh:** Script automated analyzer (`scripts/benchmark.sh` & `scripts/generate_graphs.py`) untuk me-run test matrices N ∈ {256, 512, 1024, 2048} dengan visualisasi grafik analisis performa.

---

## 💻 Konfigurasi Sistem

Untuk menjamin tingkat akurasi dan replikasi actual test results, seluruh test dijalankan pada lingkungan komputasi dengan konfigurasi terstandar berikut:

> 🔬 **Metodologi Pengukuran (Methodological Guardrail):**  
> Testing dilakukan dalam kondisi sistem *idle* (background load minimal) dengan CPU Governor disetel ke mode **'performance'** guna menjaga konsistensi frekuensi core (mencegah *frequency throttling*), serta *clock rate* GPU dipastikan stabil selama seluruh sesi benchmark berjalan untuk menjamin konsistensi test data.

<details>
<summary><b>🔍 Klik untuk melihat detail spesifikasi hardware dan software</b></summary>

* **Processor (CPU):** Intel® Core™ i5-14450HX
  * Arsitektur Hybrid: 6 Performance Cores (P-Cores) & 4 Efficient Cores (E-Cores)
  * Parallel Processing: 16 Threads, Hyper-Threading Enabled, 20 MB Intel® Smart Cache
  * Frekuensi Turbo Maksimum: 4.80 GHz
* **Graphics Card (GPU):** NVIDIA® GeForce RTX™ 4050 Laptop GPU
  * Arsitektur: Ada Lovelace (6 GB GDDR6 Dedicated VRAM)
  * Spesifikasi Compute: 2560 CUDA Cores, Dukungan FMA (Fused Multiply-Add), & Tensor Cores
  * Daya Kerja Maksimum (TGP): Up to 96W
* **Memory & Interconnect:**
  * RAM Sistem: 16 GB DDR5 Dual-Channel @ 4800 MHz
  * Interconnect: PCIe Gen 4 x8 Lane (CPU ↔ GPU Communication)
* **Operating System:** Arch Linux x86_64 (Kernel Linux 7.0.9-arch1-1)
* **Toolchain & Compiler:** GCC 16.1.1 (C11 Standard)
* **API Akselerasi CPU:** OpenMP 5.2 (Multi-threaded Parallelism)
* **API Akselerasi GPU:** OpenCL 3.0 (NVIDIA OpenCL ICD Platform)

</details>

---

## 📊 Test Results

### 🔑 Temuan Kunci (Key Findings)

* ⚡ **Crossover Point (N ≈ 600–800, antara 512 dan 1024):** GPU (OpenCL) mulai mengungguli CPU sequential pada ukuran matriks di antara N=512 (speedup 0.68×, GPU masih **lebih lambat**) dan N=1024 (speedup 22.52×). Titik pasti crossover membutuhkan data tambahan di rentang N ∈ {640, 768, 896}.
* 📈 **Percepatan Maksimum (~140x):** Pada ukuran matriks N = 2048, GPU NVIDIA RTX 4050 mengungguli CPU sequential hingga **140.01x** dan CPU paralel (OpenMP) hingga **19.48x**.
* ⚠️ **API Initialization Overhead:** Pada matriks kecil (N = 256), GPU justru lambat (0.08x dari sequential) akibat overhead tetap inisialisasi API OpenCL (~0.095–0.100s konstan) yang mendominasi >98% total waktu eksekusi.
* 🧠 **Memory-bound → Compute-bound Transition:** Bottleneck sistem bergeser dari *fixed-cost API overhead* dan bandwidth transfer PCIe (pada N kecil) ke throughput komputasi aritmatika (pada N besar).

---

### 📋 Performance Comparison Table

Berikut adalah data actual test results yang tercatat pada sistem kami (diambil dari rata-rata 3x running execution setelah 1x *warmup*):

| Ukuran Matriks (N) | Metode Execution | Waktu Rata-rata (s) ± SD | Speedup (vs Baseline) | GFLOPS (Wall-clock) | Validitas Numerik |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **N = 256** | Sequential (CPU Baseline) | 0.0089 ±0.0010 s | 1.00x *(Reference)* | 3.79 | *Reference* |
| | OpenMP (6 Threads P-Core) | 0.0025 ±0.0005 s | 3.56x | 13.22 | ✅ **VALID** |
| | OpenCL (GPU Tiled) | 0.0905 ±0.0049 s | 0.10x | 0.37 | ✅ **VALID** |
| **N = 512** | Sequential (CPU Baseline) | 0.0654 ±0.0009 s | 1.00x *(Reference)* | 4.10 | *Reference* |
| | OpenMP (6 Threads P-Core) | 0.0115 ±0.0001 s | 5.69x | 23.33 | ✅ **VALID** |
| | OpenCL (GPU Tiled) | 0.0902 ±0.0084 s | **0.73x** ⚠️ | 2.97 | ✅ **VALID** |
| **N = 1024** | Sequential (CPU Baseline) | 2.4411 ±0.0066 s | 1.00x *(Reference)* | 0.88 | *Reference* |
| | OpenMP (6 Threads P-Core) | 0.3959 ±0.0021 s | 6.17x | 5.42 | ✅ **VALID** |
| | OpenCL (GPU Tiled) | 0.1084 ±0.0067 s | 22.52x | 19.82 | ✅ **VALID** |
| **N = 2048** | Sequential (CPU Baseline) | 21.3689 ±0.0060 s | 1.00x *(Reference)* | 0.80 | *Reference* |
| | OpenMP (6 Threads P-Core) | 3.1148 ±0.0276 s | 6.86x | 5.52 | ✅ **VALID** |
| | OpenCL (GPU Tiled) | 0.1470 ±0.0092 s | 145.34x | 116.85 | ✅ **VALID** |

> 📐 **Definisi GFLOPS:** Kolom GFLOPS di atas dihitung berdasarkan **wall-clock time** (total waktu dari awal hingga akhir, termasuk overhead inisialisasi API OpenCL). Rumus: `GFLOPS = (2 × N³) / (T_total × 10⁹)`. Untuk GPU, GFLOPS berbasis waktu kernel murni ($T_{kernel}$) secara signifikan lebih tinggi — lihat [Laporan Analisis Lengkap](docs/analysis.md) §4.2 untuk breakdown.

> ⚠️ **Catatan Reproduksibilitas (Reproducibility Note):**  
> Hasil benchmark di atas dapat bervariasi bergantung pada arsitektur mikro CPU/GPU, batas daya (TDP) sistem pendingin laptop, *core temperature*, serta kondisi bandwidth bus PCIe yang digunakan selama testing.
> 
---

### 🔬 Analisis Kinerja Teoritis vs Aktual (Theoretical vs Actual Performance)

* **GPU Peak FP32 (Teoritis @ 2055 MHz):** ~10.5 TFLOPS (10,522 GFLOPS)
  * *Perhitungan: 2 (FMA) × 2560 CUDA Cores × 2055 MHz = 10,522 GFLOPS*
  * *GPU dikunci pada 2055 MHz selama benchmark; peak dihitung pada frekuensi yang sama untuk konsistensi.*
* **GPU Measured FP32 (Aktual pada N = 2048):**
  * Berbasis $T_{total}$ (wall-clock): **105.55 GFLOPS** — Efisiensi: **~1.00%**
  * Berbasis $T_{kernel}$ (komputasi murni): **~297 GFLOPS** — Efisiensi: **~2.82%**

**Analisis Celah Efisiensi:**
Meskipun pengoptimalan *Matrix Tiling* berukuran 16 × 16 pada memori lokal berhasil meningkatkan efisiensi secara signifikan dibandingkan akses memori global langsung (karena memanfaatkan cache L1/L2 GPU secara optimal), performa aktual masih jauh di bawah batas teoritis kartu grafis. Hal ini disebabkan oleh:
1. **Fixed API Initialization Overhead:** Setiap invokasi OpenCL melakukan re-inisialisasi penuh (platform discovery, context creation, program build, buffer allocation) yang menghabiskan ~0.095–0.100s konstan. Ini menekan GFLOPS wall-clock secara drastis, terutama pada matriks kecil.
2. **Sub-optimal Tiling & Hardware Alignment:** Kernel OpenCL generik tidak memiliki optimasi mikro khusus seperti *register tiling* (menyimpan data langsung di register *thread*), pemanfaatan *Tensor Cores* (melalui instruksi khusus hardware), atau optimasi assembly tingkat rendah seperti yang disediakan oleh library vendor tertutup (proprietary) seperti **NVIDIA CUDA** atau **cuBLAS**.
3. **Locked Clock Rate Limit (Penguncian Clock GPU):** Selama sesi benchmark, frekuensi clock GPU dikunci secara manual pada 2055 MHz (dibandingkan batas boost dinamis maksimumnya hingga 3105 MHz) untuk menjamin stabilitas data uji dan mencegah throttling suhu. Hal ini membatasi kapasitas throughput komputasi teoritis puncak kartu grafis selama pengujian.

---

### 💡 Analisis Ilmiah Hasil Eksperimen

* **Efek API Overhead (N=256):** Pada matriks kecil, GPU OpenCL justru **12.5× lebih lambat** dibanding CPU karena *fixed-cost* overhead inisialisasi API OpenCL (~0.099s) mendominasi 99.66% total waktu eksekusi. Waktu komputasi kernel murni hanya 0.13 ms.
* **Zona Transisi (N=512):** GPU masih **47% lebih lambat** dari CPU sequential (speedup 0.68×). Meskipun proporsi waktu kernel meningkat ke 1.00%, overhead API tetap mendominasi (98.41%). Crossover point sesungguhnya diestimasi berada pada **N ≈ 600–800**.
* **GPU Dominance (N ≥ 1024):** Pada N=1024, GPU mulai mengungguli CPU sequential sebesar 22.52×. Pada N=2048, keunggulan melonjak hingga **~140×** berkat taktik *Matrix Tiling* dan optimalisasi memori lokal. Rasio intensitas aritmatika meningkat tajam sehingga sistem bergeser menjadi **compute-bound**.
* **Penurunan Dramatis GFLOPS CPU — Analisis Cache Working Set:**
  * N=256: Working set = 3 × 256² × 4 bytes = **0.75 MB** → masuk penuh ke L2 cache → **4.43 GFLOPS**
  * N=512: Working set = **3.0 MB** → masuk L3 cache (20 MB) → **4.06 GFLOPS**
  * N=1024: Working set = **12.0 MB** → mendekati batas L3 → **0.88 GFLOPS** (terjadi *cache capacity thrashing*)
  * N=2048: Working set = **48.0 MB** → jauh melebihi L3 → **0.75 GFLOPS** (akses memori utama mendominasi)
* **Kemungkinan Akselerasi Lanjutan (OpenCL vs CUDA):** Performa GPU kemungkinan besar dapat meningkat secara signifikan jika diimplementasikan menggunakan **NVIDIA CUDA** atau **cuBLAS**, karena optimalisasi khusus-vendor yang disesuaikan dengan arsitektur GPU Ada Lovelace.

---

## 📈 Grafik Kinerja

#### 1. Execution Time Comparison (Lower is Better)
![Execution Time Comparison](test/graphs/execution_time_v2.png)
*💡 **Insight:** CPU OpenMP memimpin pada dimensi kecil (N ≤ 512), namun pada N ≥ 1024 waktu execution GPU OpenCL jauh lebih rendah karena beban transfer PCIe berhasil terkompensasi oleh kecepatan komputasi paralel.*

#### 2. Faktor Peningkatan Kinerja / Speedup (Higher is Better)
![Speedup Comparison](test/graphs/speedup_v2.png)
*💡 **Insight:** Speedup GPU melonjak secara eksponensial dari 0.08x (pada N = 256) hingga mencapai 140.01x (pada N = 2048), memvalidasi keunggulan komputasi throughput GPU pada massive workload.*

#### 3. Ringkasan Kinerja Gabungan (Log Scale)
![Combined Performance Overview](test/graphs/combined_overview_v2.png)
*💡 **Insight:** Grafik skala logaritma memperlihatkan kurva komparatif yang jelas tentang pergeseran keunggulan performa dari CPU ke GPU. Crossover point diestimasi terjadi antara N=512 (GPU masih kalah) dan N=1024 (GPU menang telak).*

#### 4. Kinerja Komputasi - GFLOPS (Higher is Better)
![GFLOPS](test/graphs/gflops_v2.png)
*💡 **Insight:** GPU mencapai kinerja masif hingga >105 GFLOPS (wall-clock) pada matriks besar. Performa kernel murni (~297 GFLOPS pada N=2048) jauh lebih tinggi setelah mengeliminasi overhead inisialisasi API.*

#### 5. Dekomposisi Waktu Execution GPU (GPU Breakdown)
![GPU Breakdown](test/graphs/gpu_breakdown_v2.png)
*💡 **Insight:** Pada dimensi matriks kecil, sebagian besar waktu dihabiskan untuk overhead inisialisasi API OpenCL (platform/context/program/buffer re-creation) dan transfer memori H2D/D2H, membuktikan bahwa utilisasi komputasi kernel murni (warna biru) baru mulai optimal pada ukuran matriks yang besar.*

#### 6. Roofline Model — Performa vs Batas Teoritis Hardware
![Roofline Model](test/graphs/roofline_v2.png)
*💡 **Insight:** Roofline chart memperlihatkan posisi setiap measurement point relatif terhadap batas teoritis hardware (peak compute ~10.5 TFLOPS @ 2055 MHz dan memory bandwidth 192 GB/s). Semua titik berada jauh di bawah garis atap (*roofline*), mengindikasikan ruang signifikan untuk optimasi kernel (register tiling, Tensor Cores, cuBLAS).*

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
Pastikan tools compilation dan library berikut telah terinstal pada sistem Anda:
```bash
# Untuk Arch Linux
sudo pacman -S base-devel opencl-headers opencl-nvidia python-matplotlib python-numpy

# Untuk Debian/Ubuntu
sudo apt-get install build-essential opencl-headers intel-opencl-icd nvidia-opencl-icd python3-matplotlib python3-numpy
```

#### 2. OS Windows
* **Operating System Windows:** Kode program ini telah dioptimasi untuk dapat berjalan di OS Windows melalui tiga metode alternatif:
* **Metode WSL2 (Sangat Direkomendasikan):** Jalankan run program di dalam kontainer Windows Subsystem for Linux (WSL2) dengan mengikuti prosedur instalasi Linux Debian/Ubuntu di atas. Library OpenCL akan otomatis terhubung ke kartu grafis host Windows.
* **Metode Native Windows (MSYS2 / MinGW-w64):**
  1. Download dan install [MSYS2](https://www.msys2.org/).
  2. Buka terminal MSYS2 UCRT64 dan jalankan perintah instalasi tools compilation:
     ```bash
     pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make mingw-w64-ucrt-x86_64-opencl
     ```
  3. Lakukan compile menggunakan perintah `mingw32-make build`.
* **Metode Microsoft Visual Studio (MSVC Compiler):**
  1. Buat proyek C++ konsol baru (*Empty Project*) pada IDE Visual Studio Anda.
  2. Masukkan seluruh file sumber dari direktori `src/` (termasuk subfolder `opencl/`) ke dalam hierarki proyek.
  3. Aktifkan dukungan parallel processing OpenMP melalui menu properties proyek: `Configuration Properties -> C/C++ -> Language -> OpenMP Support -> Yes`.
  4. Download SDK OpenCL (biasanya dibundel bersama instalasi NVIDIA CUDA Toolkit) dan hubungkan *link library* `OpenCL.lib` di bagian *linker dependencies* Visual Studio.

---

### 💻 Execution Instructions

1. **Build Source Code:**
   Lakukan compile program untuk membuat executable target `gemm_runner`:
   ```bash
   make build
   ```

2. **Jalankan Automated Testing:**
   Jalankan benchmark otomatis untuk seluruh test matrices:
   ```bash
   make benchmark
   ```
   *Hasil waktu execution mentah akan terekspor secara otomatis ke `test/execution_time.csv`.*

3. **Hasilkan Grafik Visualisasi:**
   Jalankan visualizer untuk menghasilkan grafik analisis performa:
   ```bash
   make graphs
   ```
   *Grafik output beresolusi tinggi akan tersimpan di dalam folder `test/graphs/`.*

4. **Run Seluruh Alur Kerja (All-in-One):**
   Untuk mengompilasi, menjalankan benchmark, dan menghasilkan grafik sekaligus:
   ```bash
   make all
   ```

5. **Membersihkan Proyek:**
   Untuk menghapus binary files hasil compiler dan reset file benchmark:
   ```bash
   make clean
   ```

> 📌 **Catatan Hardware-Aware untuk Run Manual (OpenMP):**
> Jika Anda ingin menjalankan binary secara manual tanpa script otomatisasi, pastikan untuk mengunci *thread* hanya pada P-Core untuk menghindari degradasi performa akibat E-Core:
> ```bash
> OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close ./gemm_runner --mode omp --size 2048
> ```

---

## 📂 Struktur Folder

```
heterogeneous-gemm/
├── Makefile                 # Otomatisasi compiler & testing
├── README.md                # Dokumentasi utama proyek UAS
├── src/                     # Seluruh C source code & OpenCL kernel
│   ├── main.c               # Driver benchmark utama
│   ├── gemm_common.h        # Memory allocation & validasi epsilon
│   ├── sequential.c         # Fungsi perkalian sequential CPU
│   ├── openmp.c             # Paralelisasi multi-core OpenMP
│   └── opencl/
│       ├── opencl_host.c    # Pipeline Host GPU OpenCL
│       └── gemm_kernel.cl   # Kernel GPU Tiling
├── scripts/                 # Otomatisasi penganalisis data
│   ├── benchmark.sh         # Pengumpul metrik dalam bentuk CSV
│   └── generate_graphs.py   # Script visualisasi performa
├── test/                    # Folder output system testing [Wajib UAS]
│   ├── execution_time.csv   # Data metrik aktual
│   └── graphs/              # Grafik visualisasi performa
└── docs/                    # File laporan ilmiah & diagram alir
    ├── analysis.md          # Laporan pembahasan komprehensif
    └── diagrams/
        ├── gemm_banner.png      # Gambar banner proyek
        └── system_flowchart.txt # Alur logika benchmark (ASCII Diagram)
```

---

## ⚠️ Batasan Sistem & Pengembangan Lanjut

Meskipun sistem benchmark ini memberikan analisis performa heterogen yang komprehensif, terdapat beberapa keterbatasan teknis yang dapat dikembangkan lebih lanjut:
1. **Re-inisialisasi API OpenCL per Invokasi:** Implementasi saat ini melakukan inisialisasi penuh OpenCL stack (platform, context, program build, buffer allocation) di setiap pemanggilan `gemm_opencl()`. Overhead tetap ~0.095–0.100s ini tidak mencerminkan skenario deployment produksi di mana inisialisasi dilakukan sekali. Pengembangan lanjutan sebaiknya memisahkan fase *init* dari fase *compute* dalam loop pengukuran.
2. **Jumlah Titik Data Terbatas:** Benchmark hanya menguji N ∈ {256, 512, 1024, 2048}. Diperlukan titik tambahan (N ∈ {384, 640, 768, 896}) di zona transisi untuk menentukan crossover point yang presisi.
3. **Penjadwalan Blok Dinamis (Block Size Auto-Tuning):** Ukuran *tiling* OpenCL saat ini dikunci secara statis pada dimensi 16 × 16. Implementasi tingkat lanjut dapat memanfaatkan mekanisme pencarian adaptif untuk mengetes Work-Group Size terbaik berdasarkan karakteristik hardware runtime.
4. **Ketiadaan Perbandingan BLAS Teroptimasi:** GPU testing belum dibandingkan dengan OpenBLAS/MKL (CPU teroptimasi) atau NVIDIA cuBLAS (GPU teroptimasi). OpenBLAS pada i5-14450HX berpotensi mencapai 50–100+ GFLOPS melalui AVX-512 dan BLAS level-3 optimization.
5. **Optimasi Vektor CPU (Explicit SIMD):** Bagian paralelisasi CPU saat ini sepenuhnya mengandalkan optimasi compiler otomatis dan pragma OpenMP, tanpa pemanfaatan instruksi vektor hardware secara eksplisit (seperti AVX2/AVX-512).
6. **Thermal Throttling Tidak Dimonitor:** Suhu CPU/GPU selama benchmark tidak dilaporkan. Benchmark sequential N=2048 memakan ~22.8 detik, cukup untuk menyebabkan thermal throttling pada laptop GPU (TGP 96W). Stabilitas clock rate GPU selama keseluruhan sesi tidak diverifikasi secara independen.
7. **Pembatasan Frekuensi Kerja GPU (Locked Clock Rate Limit):** GPU dikunci secara manual pada frekuensi 2055 MHz demi stabilitas dan konsistensi data uji. Hal ini membatasi GPU untuk beroperasi pada frekuensi boost dinamis teoritis maksimumnya (hingga 3105 MHz), sehingga persentase efisiensi riil terhadap kapasitas komputasi puncak teoritis tampak rendah (~1.00%).

---

## 📜 Lisensi & Integritas Akademik

Proyek ini disusun sepenuhnya sebagai syarat pemenuhan Ujian Akhir Semester untuk mata kuliah Arsitektur dan Sistem Komputer di Universitas Negeri Surabaya. Seluruh data benchmark yang disajikan bersifat riil dan diambil langsung dari hardware yang tertera pada spesifikasi.

Source code dan dokumentasi ini dilisensikan di bawah [MIT License](LICENSE).

---
<p align="center">
  <b>www.unesa.ac.id | "Growing with character"</b>
</p>
