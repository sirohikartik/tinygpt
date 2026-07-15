#!/bin/bash
PROMPT="James bond"
MAX_TOKENS=100
RUNS=5

ttfts=()
avg_times=()
throughputs=()

echo "Running benchmark $RUNS times..."

for i in $(seq 1 $RUNS); do
    echo "Run $i..."
    printf "$PROMPT\n$MAX_TOKENS\n" | ./a.out > run_output.txt
    
    # Use grep with a more specific pattern and avoid matching token debug lines
    ttft=$(grep "TTFT[[:space:]]*:" run_output.txt | awk '{print $3}')
    avg_time=$(grep "Avg Time Per Token" run_output.txt | awk '{print $6}')
    throughput=$(grep "Generation Throughput" run_output.txt | awk '{print $4}')
    
    echo "TTFT: $ttft, AvgTime: $avg_time, Throughput: $throughput"
    
    ttfts+=($ttft)
    avg_times+=($avg_time)
    throughputs+=($throughput)
done

sum_ttft=0
sum_avg=0
sum_tp=0

for val in "${ttfts[@]}"; do sum_ttft=$(echo "$sum_ttft + $val" | bc); done
for val in "${avg_times[@]}"; do sum_avg=$(echo "$sum_avg + $val" | bc); done
for val in "${throughputs[@]}"; do sum_tp=$(echo "$sum_tp + $val" | bc); done

echo "--------------------------------------------------"
echo "Mean Results over $RUNS runs:"
echo "Mean TTFT: $(echo "scale=4; $sum_ttft / $RUNS" | bc) ms"
echo "Mean Avg Time Per Token: $(echo "scale=4; $sum_avg / $RUNS" | bc) ms/token"
echo "Mean Generation Throughput: $(echo "scale=4; $sum_tp / $RUNS" | bc) tokens/sec"
echo "--------------------------------------------------"
rm run_output.txt
