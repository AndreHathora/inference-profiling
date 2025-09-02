import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, Any


def create_latency_comparison_plot(df: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(12, 8))

    engines = df['engine'].unique()
    metrics = ['ttft', 'tpot', 'p50_latency', 'p95_latency', 'p99_latency']

    x = np.arange(len(metrics))
    width = 0.8 / len(engines)

    for i, engine in enumerate(engines):
        engine_data = df[df['engine'] == engine]
        if not engine_data.empty:
            values = []
            for metric in metrics:
                if metric in engine_data.columns:
                    val = engine_data[metric].mean()
                    values.append(val if not np.isnan(val) else 0)

            plt.bar(x + i * width, values, width, label=engine, alpha=0.8)

    plt.xlabel('Latency Metrics')
    plt.ylabel('Time (seconds)')
    plt.title('Latency Comparison Across Engines')
    plt.xticks(x + width * (len(engines) - 1) / 2, metrics, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/latency_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_throughput_comparison_plot(df: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(10, 6))

    engines = df['engine'].unique()
    throughputs = []

    for engine in engines:
        engine_data = df[df['engine'] == engine]
        if not engine_data.empty and 'throughput' in engine_data.columns:
            throughput = engine_data['throughput'].mean()
            throughputs.append(throughput if not np.isnan(throughput) else 0)

    if throughputs:
        plt.bar(engines, throughputs, alpha=0.7)
        plt.xlabel('Engine')
        plt.ylabel('Throughput (tokens/sec)')
        plt.title('Throughput Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/throughput_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()


def create_memory_usage_plot(df: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(10, 6))

    engines = df['engine'].unique()
    memory_usage = []

    for engine in engines:
        engine_data = df[df['engine'] == engine]
        if not engine_data.empty and 'gpu_memory_used' in engine_data.columns:
            memory = engine_data['gpu_memory_used'].mean()
            memory_usage.append(memory if not np.isnan(memory) else 0)

    if memory_usage:
        plt.bar(engines, memory_usage, alpha=0.7, color='orange')
        plt.xlabel('Engine')
        plt.ylabel('GPU Memory Used (MB)')
        plt.title('GPU Memory Usage Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/memory_usage.png", dpi=300, bbox_inches='tight')
        plt.close()


def create_length_vs_latency_plot(df: pd.DataFrame, output_dir: str):
    length_data = df[df['result_type'] == 'length_analysis'].copy()
    if length_data.empty:
        return

    plt.figure(figsize=(12, 8))

    engines = length_data['engine'].unique()

    for engine in engines:
        engine_length_data = length_data[length_data['engine'] == engine]
        if not engine_length_data.empty:
            plt.plot(engine_length_data['input_length'], engine_length_data['ttft'],
                    marker='o', label=f'{engine} - TTFT', linewidth=2)
            plt.plot(engine_length_data['input_length'], engine_length_data['tpot'],
                    marker='s', label=f'{engine} - TPOT', linewidth=2, linestyle='--')

    plt.xlabel('Input Length (tokens)')
    plt.ylabel('Latency (seconds)')
    plt.title('Input Length vs Latency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/length_vs_latency.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_concurrent_performance_plot(df: pd.DataFrame, output_dir: str):
    concurrent_data = df[df['result_type'] == 'concurrent'].copy()
    if concurrent_data.empty:
        return

    plt.figure(figsize=(12, 8))

    engines = concurrent_data['engine'].unique()

    for engine in engines:
        engine_data = concurrent_data[concurrent_data['engine'] == engine]
        if not engine_data.empty:
            plt.plot(engine_data['batch_index'], engine_data['throughput'],
                    marker='o', label=f'{engine} - Throughput', linewidth=2)
            plt.plot(engine_data['batch_index'], engine_data['ttft'],
                    marker='s', label=f'{engine} - TTFT', linewidth=2, linestyle='--')

    plt.xlabel('Batch Index')
    plt.ylabel('Performance Metric')
    plt.title('Concurrent Request Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/concurrent_performance.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_resource_utilization_plot(df: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(15, 5))

    engines = df['engine'].unique()

    gpu_util = []
    cpu_util = []
    engine_labels = []

    for engine in engines:
        engine_data = df[df['engine'] == engine]
        if not engine_data.empty:
            if 'gpu_utilization' in engine_data.columns:
                gpu_util.append(engine_data['gpu_utilization'].mean())
            else:
                gpu_util.append(0)

            if 'cpu_utilization' in engine_data.columns:
                cpu_util.append(engine_data['cpu_utilization'].mean())
            else:
                cpu_util.append(0)

            engine_labels.append(engine)

    x = np.arange(len(engine_labels))
    width = 0.35

    plt.subplot(1, 2, 1)
    plt.bar(x, gpu_util, width, alpha=0.7, color='red', label='GPU')
    plt.xlabel('Engine')
    plt.ylabel('GPU Utilization (%)')
    plt.title('GPU Utilization')
    plt.xticks(x, engine_labels, rotation=45)

    plt.subplot(1, 2, 2)
    plt.bar(x, cpu_util, width, alpha=0.7, color='blue', label='CPU')
    plt.xlabel('Engine')
    plt.ylabel('CPU Utilization (%)')
    plt.title('CPU Utilization')
    plt.xticks(x, engine_labels, rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/resource_utilization.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_phase_breakdown_plot(df: pd.DataFrame, output_dir: str):
    phase_data = df[df['result_type'] == 'overall'].copy()
    if phase_data.empty:
        return

    for _, row in phase_data.iterrows():
        if 'phase_breakdown' in row:
            phases = row['phase_breakdown']
            if isinstance(phases, dict):
                plt.figure(figsize=(10, 6))

                phase_names = list(phases.keys())
                phase_values = list(phases.values())

                plt.pie(phase_values, labels=phase_names, autopct='%1.1f%%', startangle=90)
                plt.title(f'Phase Breakdown - {row.get("engine", "Unknown")}')
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/phase_breakdown_{row.get('engine', 'unknown')}.png",
                           dpi=300, bbox_inches='tight')
                plt.close()


def generate_all_plots(df: pd.DataFrame, output_dir: str = './plots'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating comparison plots...")

    try:
        create_latency_comparison_plot(df, output_dir)
        print("Latency comparison plot created")
    except Exception as e:
        print(f"Latency comparison plot failed: {e}")

    try:
        create_throughput_comparison_plot(df, output_dir)
        print("Throughput comparison plot created")
    except Exception as e:
        print(f"Throughput comparison plot failed: {e}")

    try:
        create_memory_usage_plot(df, output_dir)
        print("Memory usage plot created")
    except Exception as e:
        print(f"Memory usage plot failed: {e}")

    try:
        create_length_vs_latency_plot(df, output_dir)
        print("Length vs latency plot created")
    except Exception as e:
        print(f"Length vs latency plot failed: {e}")

    try:
        create_concurrent_performance_plot(df, output_dir)
        print("Concurrent performance plot created")
    except Exception as e:
        print(f"Concurrent performance plot failed: {e}")

    try:
        create_resource_utilization_plot(df, output_dir)
        print("Resource utilization plot created")
    except Exception as e:
        print(f"Resource utilization plot failed: {e}")

    try:
        create_phase_breakdown_plot(df, output_dir)
        print("Phase breakdown plots created")
    except Exception as e:
        print(f"Phase breakdown plots failed: {e}")

    print(f"\nAll plots saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate plots from benchmark results')
    parser.add_argument('--input', type=str, default='./processed_results/processed_metrics.csv',
                       help='Input CSV file with processed metrics')
    parser.add_argument('--output', type=str, default='./plots',
                       help='Output directory for plots')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Input file not found: {args.input}")
        print("Run metrics_collector.py first to process the data")
        exit(1)

    print(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)

    print(f"Loaded {len(df)} rows of data")
    print(f"Columns: {list(df.columns)}")

    generate_all_plots(df, args.output)
