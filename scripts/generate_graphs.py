#!/usr/bin/env python3
import csv, os, sys, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH  = "results/execution_time.csv"
GRAPH_DIR = "results/graphs"
TEST_GRAPH_DIR = "test/graphs"
DPI = 180

BG_COLOR = "#0B0F19"
AX_COLOR = "#111827"
TEXT_COLOR = "#F3F4F6"
GRID_COLOR = "#1F2937"

COLORS = {"seq": "#F43F5E", "omp": "#10B981", "opencl": "#3B82F6"}
LABELS = {"seq": "Sequential (CPU)", "omp": "OpenMP (CPU Parallel)", "opencl": "OpenCL (GPU)"}

def apply_dark_theme(fig, ax):
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(AX_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['top'].set_color(GRID_COLOR)
    ax.spines['right'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, which='both')
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.5, zorder=0)

def load_csv(path):
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data.setdefault(row["mode"], {})[int(row["size"])] = float(row["time"])
    return data

def plot_execution_time(data, sizes):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    x, w = np.arange(len(sizes)), 0.25
    for i, m in enumerate(["seq", "omp", "opencl"]):
        if m not in data: continue
        t = [data[m].get(s, 0) for s in sizes]
        bars = ax.bar(x + i*w, t, w, label=LABELS[m], color=COLORS[m], edgecolor=BG_COLOR, linewidth=1, zorder=3)
        for b, v in zip(bars, t):
            ax.text(b.get_x()+b.get_width()/2, b.get_height() + (v*0.01 if v > 0.1 else 0.002), 
                    f"{v:.4f}s", ha="center", va="bottom", fontsize=8, color=TEXT_COLOR, fontweight="bold")
    ax.set_xlabel("Matrix Size (N×N)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Execution Time (seconds)", fontsize=11, fontweight="bold")
    ax.set_title("GEMM Execution Time Comparison (Lower is Better)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x+w); ax.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax.legend(facecolor=AX_COLOR, edgecolor=GRID_COLOR, loc="upper left")
    apply_dark_theme(fig, ax)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "execution_time.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    fig.savefig(os.path.join(TEST_GRAPH_DIR, "execution_time.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)

def plot_speedup(data, sizes):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    for m in ["omp", "opencl"]:
        if m not in data or "seq" not in data: continue
        sp = [data["seq"].get(s,1)/max(data[m].get(s,1),1e-9) for s in sizes]
        ax.plot(sizes, sp, marker="o", lw=3, ms=8, label=LABELS[m], color=COLORS[m], zorder=3)
        for s, v in zip(sizes, sp):
            ax.annotate(f"{v:.1f}x", (s, v), textcoords="offset points", xytext=(0,10), 
                        ha="center", fontsize=9, fontweight="bold", color=COLORS[m])
    ax.axhline(y=1, color="#94A3B8", ls="--", lw=1.5, label="CPU Baseline", zorder=2)
    ax.set_xlabel("Matrix Size (N)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Speedup factor (vs Sequential)", fontsize=11, fontweight="bold")
    ax.set_title("Performance Speedup (Higher is Better)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(sizes)
    ax.legend(facecolor=AX_COLOR, edgecolor=GRID_COLOR, loc="upper left")
    apply_dark_theme(fig, ax)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "speedup.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    fig.savefig(os.path.join(TEST_GRAPH_DIR, "speedup.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)

def plot_combined(data, sizes):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor=BG_COLOR)
    
    for m in ["seq","omp","opencl"]:
        if m not in data: continue
        a1.plot(sizes, [data[m].get(s,0) for s in sizes], marker="s", lw=2.5, ms=6, label=LABELS[m], color=COLORS[m], zorder=3)
    a1.set_yscale("log")
    a1.set_xlabel("Matrix Size (N)", fontsize=11, fontweight="bold")
    a1.set_ylabel("Time (seconds, log scale)", fontsize=11, fontweight="bold")
    a1.set_title("Execution Time (Log Scale)", fontsize=12, fontweight="bold")
    a1.set_xticks(sizes)
    a1.legend(facecolor=AX_COLOR, edgecolor=GRID_COLOR, loc="lower left")
    apply_dark_theme(fig, a1)
    
    for m in ["omp","opencl"]:
        if m not in data or "seq" not in data: continue
        a2.plot(sizes, [data["seq"].get(s,1)/max(data[m].get(s,1),1e-9) for s in sizes], marker="o", lw=2.5, ms=6, label=LABELS[m], color=COLORS[m], zorder=3)
    a2.axhline(y=1, color="#94A3B8", ls="--", lw=1.5, label="CPU Baseline", zorder=2)
    a2.set_xlabel("Matrix Size (N)", fontsize=11, fontweight="bold")
    a2.set_ylabel("Speedup factor", fontsize=11, fontweight="bold")
    a2.set_title("Speedup vs CPU Baseline", fontsize=12, fontweight="bold")
    a2.set_xticks(sizes)
    a2.legend(facecolor=AX_COLOR, edgecolor=GRID_COLOR, loc="upper left")
    apply_dark_theme(fig, a2)
    
    fig.suptitle("Heterogeneous GEMM Performance Analysis (CPU vs GPU)", fontsize=15, fontweight="bold", color=TEXT_COLOR, y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "combined_overview.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    fig.savefig(os.path.join(TEST_GRAPH_DIR, "combined_overview.png"), dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)

def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(1)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(TEST_GRAPH_DIR, exist_ok=True)
    data = load_csv(CSV_PATH)
    sizes = sorted({s for d in data.values() for s in d})
    plot_execution_time(data, sizes)
    plot_speedup(data, sizes)
    plot_combined(data, sizes)
    shutil.copy2(CSV_PATH, "test/execution_time.csv")

if __name__ == "__main__":
    main()
