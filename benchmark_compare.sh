#!/bin/bash
PROMPT="James bond"
MAX_TOKENS=50
RUNS=3

echo "=================================================="
echo "          TINYGPT CPU vs MPS BENCHMARK            "
echo "=================================================="
echo "Prompt: '$PROMPT' | Max Tokens: $MAX_TOKENS | Runs: $RUNS"
echo ""

# 1. Benchmark CPU
echo ">>> Building CPU target (make cpu)..."
make cpu > /dev/null 2>&1

ttfts_cpu=()
avg_times_cpu=()
throughputs_cpu=()

echo ">>> Running CPU benchmarks ($RUNS runs)..."
for i in $(seq 1 $RUNS); do
    printf "$PROMPT\n$MAX_TOKENS\n" | ./a.out > run_output_cpu.txt
    ttft=$(grep "TTFT[[:space:]]*:" run_output_cpu.txt | awk '{print $3}')
    avg_time=$(grep "Avg Time Per Token" run_output_cpu.txt | awk '{print $6}')
    throughput=$(grep "Generation Throughput" run_output_cpu.txt | awk '{print $4}')
    echo "  [CPU Run $i] TTFT: ${ttft} ms | Avg Time: ${avg_time} ms/token | Throughput: ${throughput} tok/s"
    ttfts_cpu+=($ttft)
    avg_times_cpu+=($avg_time)
    throughputs_cpu+=($throughput)
done

# 2. Benchmark MPS
echo ""
echo ">>> Building MPS / Metal GPU target (make mps)..."
make mps > /dev/null 2>&1

ttfts_mps=()
avg_times_mps=()
throughputs_mps=()

echo ">>> Running MPS benchmarks ($RUNS runs)..."
for i in $(seq 1 $RUNS); do
    printf "$PROMPT\n$MAX_TOKENS\n" | ./a.out > run_output_mps.txt
    ttft=$(grep "TTFT[[:space:]]*:" run_output_mps.txt | awk '{print $3}')
    avg_time=$(grep "Avg Time Per Token" run_output_mps.txt | awk '{print $6}')
    throughput=$(grep "Generation Throughput" run_output_mps.txt | awk '{print $4}')
    echo "  [MPS Run $i] TTFT: ${ttft} ms | Avg Time: ${avg_time} ms/token | Throughput: ${throughput} tok/s"
    ttfts_mps+=($ttft)
    avg_times_mps+=($avg_time)
    throughputs_mps+=($throughput)
done

# Calculate CPU Means
sum_ttft_cpu=0; sum_avg_cpu=0; sum_tp_cpu=0
for val in "${ttfts_cpu[@]}"; do sum_ttft_cpu=$(echo "$sum_ttft_cpu + $val" | bc); done
for val in "${avg_times_cpu[@]}"; do sum_avg_cpu=$(echo "$sum_avg_cpu + $val" | bc); done
for val in "${throughputs_cpu[@]}"; do sum_tp_cpu=$(echo "$sum_tp_cpu + $val" | bc); done

mean_ttft_cpu=$(echo "scale=2; $sum_ttft_cpu / $RUNS" | bc)
mean_avg_cpu=$(echo "scale=2; $sum_avg_cpu / $RUNS" | bc)
mean_tp_cpu=$(echo "scale=2; $sum_tp_cpu / $RUNS" | bc)

# Calculate MPS Means
sum_ttft_mps=0; sum_avg_mps=0; sum_tp_mps=0
for val in "${ttfts_mps[@]}"; do sum_ttft_mps=$(echo "$sum_ttft_mps + $val" | bc); done
for val in "${avg_times_mps[@]}"; do sum_avg_mps=$(echo "$sum_avg_mps + $val" | bc); done
for val in "${throughputs_mps[@]}"; do sum_tp_mps=$(echo "$sum_tp_mps + $val" | bc); done

mean_ttft_mps=$(echo "scale=2; $sum_ttft_mps / $RUNS" | bc)
mean_avg_mps=$(echo "scale=2; $sum_avg_mps / $RUNS" | bc)
mean_tp_mps=$(echo "scale=2; $sum_tp_mps / $RUNS" | bc)

echo ""
echo "=================================================="
echo "                 BENCHMARK SUMMARY                "
echo "=================================================="
printf "%-25s | %-15s | %-15s\n" "Metric" "CPU (Accelerate)" "MPS (Metal GPU)"
echo "-----------------------------------------------------------------"
printf "%-25s | %-12s ms | %-12s ms\n" "Mean TTFT" "$mean_ttft_cpu" "$mean_ttft_mps"
printf "%-25s | %-9s ms/tok | %-9s ms/tok\n" "Mean Decode Time" "$mean_avg_cpu" "$mean_avg_mps"
printf "%-25s | %-10s tok/s | %-10s tok/s\n" "Mean Throughput" "$mean_tp_cpu" "$mean_tp_mps"
echo "=================================================="

rm -f run_output_cpu.txt run_output_mps.txt
