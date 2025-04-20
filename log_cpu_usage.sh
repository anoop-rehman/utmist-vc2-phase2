#!/bin/bash

# Define output files
OUTPUT_FILE="cpu_usage_log5.csv"
PROCESSES_FILE="top_processes_log5.txt"

# Create CSV with headers
echo "timestamp,user_pct,system_pct,idle_pct,load_avg_1m,load_avg_5m,load_avg_15m" > $OUTPUT_FILE
> $PROCESSES_FILE  # Initialize empty file

# Duration in seconds (15 minutes = 900 seconds)
DURATION=900

echo "Logging CPU usage for $DURATION seconds to $OUTPUT_FILE..."
echo "Top 10 processes with detailed stats will be logged to $PROCESSES_FILE every second"
echo "Press Ctrl+C to stop early"

# Start logging loop
for ((i=1; i<=$DURATION; i++)); do
    # Get timestamp
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    
    # Get CPU usage with top (one iteration, batch mode)
    TOP_OUTPUT=$(top -b -n 1)
    
    # Extract CPU percentages using grep and awk
    CPU_LINE=$(echo "$TOP_OUTPUT" | grep -m 1 "%Cpu(s)")
    USER_PCT=$(echo "$CPU_LINE" | awk '{print $2}')
    SYSTEM_PCT=$(echo "$CPU_LINE" | awk '{print $4}')
    IDLE_PCT=$(echo "$CPU_LINE" | awk '{print $8}')
    
    # Extract load average
    LOAD_LINE=$(echo "$TOP_OUTPUT" | grep -m 1 "load average")
    LOAD_1M=$(echo "$LOAD_LINE" | awk '{print $12}' | tr -d ',')
    LOAD_5M=$(echo "$LOAD_LINE" | awk '{print $13}' | tr -d ',')
    LOAD_15M=$(echo "$LOAD_LINE" | awk '{print $14}')
    
    # Write to CSV
    echo "$TIMESTAMP,$USER_PCT,$SYSTEM_PCT,$IDLE_PCT,$LOAD_1M,$LOAD_5M,$LOAD_15M" >> $OUTPUT_FILE
    
    # Log top 10 processes by CPU usage with detailed information (every second)
    echo -e "\n===== Top 10 Processes at $TIMESTAMP =====" >> $PROCESSES_FILE
    echo -e "PID\tUSER\t\tPR\tNI\tVIRT\t\tRES\t\tSHR\t\tS\t%CPU\t%MEM\tTIME+\t\tCOMMAND" >> $PROCESSES_FILE
    
    # Get detailed process info with custom formatted output
    # Using ps command for more detailed and controllable output than top
    ps aux --sort=-%cpu | head -11 | tail -10 | awk '{printf "%-8s\t%-12s\t%-4s\t%-4s\t%-10s\t%-10s\t%-10s\t%-4s\t%-6s\t%-6s\t%-12s\t%s\n", $2, $1, $3, $4, $5, $6, $7, $8, $3, $4, $10, $11}' >> $PROCESSES_FILE
    
    # Add more detailed memory information for top processes
    echo -e "\nDetailed Memory Usage (RSS):" >> $PROCESSES_FILE
    ps -eo pid,pmem,rss,vsz,comm --sort=-rss | head -11 | tail -10 >> $PROCESSES_FILE
    
    # Display progress
    if [ $((i % 10)) -eq 0 ]; then
        echo "Logged $i seconds... (CPU: ${USER_PCT}% user, ${SYSTEM_PCT}% system, ${IDLE_PCT}% idle)"
    fi
    
    # Wait for 1 second
    sleep 1
done

echo "Logging complete. Data saved to $OUTPUT_FILE and detailed process information to $PROCESSES_FILE" 