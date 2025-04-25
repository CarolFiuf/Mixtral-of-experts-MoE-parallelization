You can only modify `csrc` folder

# Build

```bash
srun --time=05:00 --pty --gres=gpu:0 python3 setup.py develop --install-dir ./install
```

# Test & Benchmark

```bash
srun --time=05:00 --pty --gres=gpu:1 python3 test_moe.py # For testing and benchmarking
TEST_ONLY=1 srun --time=05:00 --pty --gres=gpu:1 python3 test_moe.py # For testing
BENCHMARK_ONLY=1 srun --time=05:00 --pty --gres=gpu:1 python3 test_moe.py # For benchmarking
```