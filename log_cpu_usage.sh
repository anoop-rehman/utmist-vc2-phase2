#!/bin/bash

# Define output file
OUTPUT_FILE="cpu_usage_log.csv"

# Create CSV with headers
echo "timestamp,user_pct,system_pct,idle_pct,load_avg_1m,load_avg_5m,load_avg_15m" > $OUTPUT_FILE

# Duration in seconds (15 minutes = 900 seconds)
DURATION=900

echo "Logging CPU usage for $DURATION seconds to $OUTPUT_FILE..."
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
    
    # Display progress
    if [ $((i % 10)) -eq 0 ]; then
        echo "Logged $i seconds... (CPU: ${USER_PCT}% user, ${SYSTEM_PCT}% system, ${IDLE_PCT}% idle)"
    fi
    
    # Wait for 1 second
    sleep 1
done

echo "Logging complete. Data saved to $OUTPUT_FILE" 