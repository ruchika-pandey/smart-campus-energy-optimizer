# models/benchmark.py
import onnxruntime as ort
import numpy as np
import time
import psutil
import mlflow
from onnx import helper, TensorProto, checker, shape_inference

def create_simple_model():
    """
    Create a minimal ONNX model: single Gemm layer from 3 inputs to 1 output.
    Weight shape (3,1), bias shape (1,).
    """
    # Input: [batch, 3]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, 3])
    
    # Weight: [3, 1]
    W = helper.make_tensor('W', TensorProto.FLOAT, [3, 1],
                           np.random.randn(3, 1).astype(np.float32).flatten())
    # Bias: [1]
    B = helper.make_tensor('B', TensorProto.FLOAT, [1],
                           np.random.randn(1).astype(np.float32))
    
    # Gemm node: Y = alpha * A * B + beta * C
    # Here A = X (Mx3), B = W (3x1), C = B (1,) -> broadcasts to Mx1
    gemm = helper.make_node('Gemm', ['X', 'W', 'B'], ['Y'], alpha=1.0, beta=1.0)
    
    # Output: [batch, 1]
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, 1])
    
    graph = helper.make_graph([gemm], 'simple_model', [X], [Y], initializer=[W, B])
    model = helper.make_model(graph, producer_name='benchmark')
    model = shape_inference.infer_shapes(model)
    checker.check_model(model)
    return model

def run_benchmark(model, provider, n_iterations=5000, batch_size=1):
    """Run inference benchmark with optimizations disabled."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), sess_options, providers=[provider])
    
    input_data = np.random.randn(batch_size, 3).astype(np.float32)
    # Warm-up
    for _ in range(100):
        sess.run(['Y'], {'X': input_data})
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        sess.run(['Y'], {'X': input_data})
    end = time.perf_counter()
    
    avg_ms = ((end - start) / n_iterations) * 1000
    return avg_ms

def main():
    print("="*60)
    print("🔬 CPU vs GPU Inference Benchmark (ONNX Runtime)")
    print("="*60)
    
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    
    # System info
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()
    print(f"CPU cores: {cpu_count}")
    if cpu_freq:
        print(f"CPU frequency: {cpu_freq.current:.0f} MHz")
    print(f"RAM: {mem.total / (1024**3):.1f} GB")
    
    print("\n📦 Creating minimal ONNX model...")
    model = create_simple_model()
    
    test_providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in providers:
        test_providers.append('CUDAExecutionProvider')
    elif 'ROCMExecutionProvider' in providers:
        test_providers.append('ROCMExecutionProvider')
    
    results = {}
    for provider in test_providers:
        print(f"\n⚙️  Benchmarking on {provider}...")
        try:
            avg_ms = run_benchmark(model, provider)
            results[provider] = avg_ms
            print(f"   ✅ Average inference time: {avg_ms:.3f} ms")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if len(results) > 1:
        cpu_time = results.get('CPUExecutionProvider')
        gpu_provider = [p for p in test_providers if p != 'CPUExecutionProvider'][0]
        gpu_time = results.get(gpu_provider)
        if cpu_time and gpu_time:
            speedup = cpu_time / gpu_time
            print("\n" + "="*60)
            print(f"🚀 Speedup (CPU / {gpu_provider}): {speedup:.2f}x")
            print("="*60)
    
    # Log to MLflow
    with mlflow.start_run(run_name="ONNX_CPU_GPU_Benchmark"):
        mlflow.log_param("cpu_cores", cpu_count)
        mlflow.log_param("ram_gb", round(mem.total / (1024**3), 1))
        if cpu_freq:
            mlflow.log_param("cpu_freq_mhz", cpu_freq.current)
        mlflow.log_param("available_providers", str(providers))
        
        for provider, avg_ms in results.items():
            name = provider.replace('ExecutionProvider', '')
            mlflow.log_metric(f"latency_ms_{name}", avg_ms)
        
        if len(results) > 1 and cpu_time and gpu_time:
            mlflow.log_metric("speedup_cpu_vs_gpu", speedup)
    
    print("\n✅ Benchmark complete. Results logged to MLflow.")
    print("Run 'mlflow ui' to view them.\n")

if __name__ == "__main__":
    main()