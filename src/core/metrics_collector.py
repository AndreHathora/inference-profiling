import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import glob

def load_benchmark_results(results_dir: str) -> Dict[str, Any]:
    results = {}
    results_path = Path(results_dir)
    for json_file in results_path.glob("*_results.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results[json_file.stem] = convert_metrics_to_dict(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return results

def convert_metrics_to_dict(data):
    if isinstance(data, dict):
        converted = {}
        for key, value in data.items():
            if key == 'metrics' and isinstance(value, list):
                converted[key] = [parse_inference_metrics_string(item) for item in value]
            elif key == 'overall_metrics' and isinstance(value, str):
                converted[key] = parse_inference_metrics_string(value)
            elif key == 'concurrent_metrics' and isinstance(value, list):
                converted[key] = [parse_inference_metrics_string(item) for item in value]
            elif key == 'length_analysis' and isinstance(value, list):
                converted[key] = []
                for item in value:
                    if isinstance(item, dict) and 'metrics' in item:
                        item_copy = item.copy()
                        if isinstance(item_copy['metrics'], str):
                            item_copy['metrics'] = parse_inference_metrics_string(item_copy['metrics'])
                        converted[key].append(item_copy)
            else:
                converted[key] = convert_metrics_to_dict(value) if isinstance(value, dict) else value
        return converted
    return data

def parse_inference_metrics_string(metrics_str):
    if not isinstance(metrics_str, str) or not metrics_str.startswith('InferenceMetrics('):
        return metrics_str
    content = metrics_str[17:-1]
    metrics_dict = {}
    for pair in content.split(', '):
        if '=' in pair:
            key, value = pair.split('=', 1)
            key = key.strip('()')
            if value.replace('.', '').replace('-', '').isdigit():
                metrics_dict[key] = float(value) if '.' in value else int(float(value))
            else:
                metrics_dict[key] = value.strip("'\"")
    return metrics_dict

def extract_metrics(results: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for result_name, result_data in results.items():
        if 'vllm' in result_name:
            engine = 'vllm'
        elif 'sglang' in result_name:
            engine = 'sglang'
        elif 'mlc' in result_name:
            engine = 'mlc'
        elif 'transformers' in result_name:
            engine = 'transformers'
        elif 'comparison' in result_name:
            continue  # Skip comparison results as they are not individual engine results
        else:
            engine = 'unknown'
        model = result_data.get('model', 'unknown')
        if model == 'unknown' and 'standard' in result_data:
            model = result_data['standard'].get('model', 'unknown')
        # Handle overall metrics directly
        if 'overall_metrics' in result_data:
            metrics_str = result_data['overall_metrics']
            if isinstance(metrics_str, str):
                metrics = parse_inference_metrics_string(metrics_str)
            else:
                metrics = metrics_str

            rows.append({
                'engine': engine,
                'model': model,
                'result_type': 'overall',
                'ttft': metrics.get('ttft', 0),
                'tpot': metrics.get('tpot', 0),
                'throughput': metrics.get('throughput', 0),
                'total_tokens': metrics.get('total_tokens', 0),
                'input_tokens': metrics.get('input_tokens', 0),
                'output_tokens': metrics.get('output_tokens', 0),
                'p50_latency': metrics.get('p50_latency', 0),
                'p95_latency': metrics.get('p95_latency', 0),
                'p99_latency': metrics.get('p99_latency', 0),
                'gpu_memory_used': metrics.get('gpu_memory_used', 0),
                'gpu_utilization': metrics.get('gpu_utilization', 0),
                'cpu_utilization': metrics.get('cpu_utilization', 0)
            })

        # Handle individual run metrics
        if 'metrics' in result_data and isinstance(result_data['metrics'], list):
            for i, metrics_item in enumerate(result_data['metrics']):
                if isinstance(metrics_item, str):
                    metrics = parse_inference_metrics_string(metrics_item)
                elif isinstance(metrics_item, dict):
                    metrics = metrics_item
                else:
                    continue

                rows.append({
                    'engine': engine,
                    'model': model,
                    'result_type': f'run_{i}',
                    'ttft': metrics.get('ttft', 0),
                    'tpot': metrics.get('tpot', 0),
                    'throughput': metrics.get('throughput', 0),
                    'total_tokens': metrics.get('total_tokens', 0),
                    'input_tokens': metrics.get('input_tokens', 0),
                    'output_tokens': metrics.get('output_tokens', 0),
                    'p50_latency': metrics.get('p50_latency', 0),
                    'p95_latency': metrics.get('p95_latency', 0),
                    'p99_latency': metrics.get('p99_latency', 0),
                    'gpu_memory_used': metrics.get('gpu_memory_used', 0),
                    'gpu_utilization': metrics.get('gpu_utilization', 0),
                    'cpu_utilization': metrics.get('cpu_utilization', 0)
                })

        # Handle legacy format
        if 'standard' in result_data and 'overall_metrics' in result_data['standard']:
            metrics = result_data['standard']['overall_metrics']
            rows.append({
                'engine': engine,
                'model': model,
                'result_type': 'overall',
                'ttft': metrics.get('ttft', 0),
                'tpot': metrics.get('tpot', 0),
                'throughput': metrics.get('throughput', 0),
                'total_tokens': metrics.get('total_tokens', 0),
                'input_tokens': metrics.get('input_tokens', 0),
                'output_tokens': metrics.get('output_tokens', 0),
                'p50_latency': metrics.get('p50_latency', 0),
                'p95_latency': metrics.get('p95_latency', 0),
                'p99_latency': metrics.get('p99_latency', 0),
                'gpu_memory_used': metrics.get('gpu_memory_used', 0),
                'gpu_utilization': metrics.get('gpu_utilization', 0),
                'cpu_utilization': metrics.get('cpu_utilization', 0)
            })
        if 'comparison' in result_data:
            for eng, comp_data in result_data['comparison'].items():
                if comp_data.get('status') == 'failed':
                    # Handle failed comparisons
                    rows.append({
                        'engine': eng,
                        'model': model,
                        'result_type': 'comparison',
                        'ttft': 0,
                        'throughput': 0,
                        'gpu_memory_used': 0,
                        'tpot': 0,
                        'total_tokens': 0,
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'p50_latency': 0,
                        'p95_latency': 0,
                        'p99_latency': 0,
                        'gpu_utilization': 0,
                        'cpu_utilization': 0
                    })
                else:
                    # Handle successful comparisons
                    rows.append({
                        'engine': eng,
                        'model': model,
                        'result_type': 'comparison',
                        'ttft': comp_data.get('latency', 0),
                        'throughput': comp_data.get('throughput', 0),
                        'gpu_memory_used': comp_data.get('gpu_memory_used', 0),
                        'tpot': 0,
                        'total_tokens': comp_data.get('output_length', 0),
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'p50_latency': 0,
                        'p95_latency': 0,
                        'p99_latency': 0,
                        'gpu_utilization': 0,
                        'cpu_utilization': 0
                    })
        if 'length_analysis' in result_data and isinstance(result_data['length_analysis'], list):
            for length_data in result_data['length_analysis']:
                if isinstance(length_data, dict) and 'metrics' in length_data:
                    metrics = length_data['metrics']
                    if isinstance(metrics, dict):
                        rows.append({
                            'engine': engine,
                            'model': model,
                            'result_type': 'length_analysis',
                            'input_length': length_data.get('input_length', 0),
                            'ttft': metrics.get('ttft', 0),
                            'tpot': metrics.get('tpot', 0),
                            'throughput': metrics.get('throughput', 0),
                            'total_tokens': metrics.get('total_tokens', 0),
                            'input_tokens': metrics.get('input_tokens', 0),
                            'output_tokens': metrics.get('output_tokens', 0),
                            'p50_latency': metrics.get('p50_latency', 0),
                            'p95_latency': metrics.get('p95_latency', 0),
                            'p99_latency': metrics.get('p99_latency', 0),
                            'gpu_memory_used': metrics.get('gpu_memory_used', 0),
                            'gpu_utilization': metrics.get('gpu_utilization', 0),
                            'cpu_utilization': metrics.get('cpu_utilization', 0)
                        })
        if 'concurrent' in result_data and 'concurrent_metrics' in result_data['concurrent']:
            for i, metrics in enumerate(result_data['concurrent']['concurrent_metrics']):
                rows.append({
                    'engine': engine,
                    'model': model,
                    'result_type': 'concurrent',
                    'batch_index': i,
                    'ttft': metrics.get('ttft', 0),
                    'tpot': metrics.get('tpot', 0),
                    'throughput': metrics.get('throughput', 0),
                    'total_tokens': metrics.get('total_tokens', 0),
                    'input_tokens': metrics.get('input_tokens', 0),
                    'output_tokens': metrics.get('output_tokens', 0),
                    'p50_latency': metrics.get('p50_latency', 0),
                    'p95_latency': metrics.get('p95_latency', 0),
                    'p99_latency': metrics.get('p99_latency', 0),
                    'gpu_memory_used': metrics.get('gpu_memory_used', 0),
                    'gpu_utilization': metrics.get('gpu_utilization', 0),
                    'cpu_utilization': metrics.get('cpu_utilization', 0)
                })
    return pd.DataFrame(rows)

def aggregate_results(df: pd.DataFrame) -> Dict[str, Any]:
    summary = {}
    if not df.empty:
        summary['overall_stats'] = {
            'total_experiments': len(df),
            'engines_tested': df['engine'].unique().tolist(),
            'models_tested': df['model'].unique().tolist()
        }
        for engine in df['engine'].unique():
            engine_data = df[df['engine'] == engine]
            summary[f'{engine}_stats'] = {
                'avg_ttft': engine_data['ttft'].mean(),
                'avg_throughput': engine_data['throughput'].mean(),
                'avg_gpu_memory': engine_data['gpu_memory_used'].mean(),
                'max_throughput': engine_data['throughput'].max(),
                'min_ttft': engine_data['ttft'].min()
            }
        if 'comparison' in df['result_type'].values:
            comp_data = df[df['result_type'] == 'comparison']
            summary['engine_comparison'] = {}
            for metric in ['ttft', 'throughput', 'gpu_memory']:
                if metric in comp_data.columns:
                    summary['engine_comparison'][metric] = {}
                    for engine in comp_data['engine'].unique():
                        engine_metric = comp_data[comp_data['engine'] == engine][metric].mean()
                        summary['engine_comparison'][metric][engine] = engine_metric
    return summary

def save_processed_data(df: pd.DataFrame, summary: Dict[str, Any], output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / 'processed_metrics.csv', index=False)
    with open(output_path / 'summary_stats.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Processed data saved to {output_path}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Summary stats: {len(summary)} categories")

def collect_and_process(results_dir: str, output_dir: str = './processed_results'):
    print(f"Loading results from {results_dir}")
    results = load_benchmark_results(results_dir)
    if not results:
        print("No results found!")
        return None, None
    print(f"Found {len(results)} result files")
    df = extract_metrics(results)
    summary = aggregate_results(df)
    save_processed_data(df, summary, output_dir)
    return df, summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process benchmark results')
    parser.add_argument('--input', type=str, default='./benchmark_results',
                       help='Input directory with benchmark results')
    parser.add_argument('--output', type=str, default='./processed_results',
                       help='Output directory for processed data')
    args = parser.parse_args()
    df, summary = collect_and_process(args.input, args.output)
    if df is not None:
        print("\nData Overview:")
        print(df.head())
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nEngines: {df['engine'].unique()}")
        print(f"\nResult types: {df['result_type'].unique()}")
