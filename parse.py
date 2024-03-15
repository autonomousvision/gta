path = "outputs/RoPE/log.txt"

# ファイルを読み込んで処理
with open(path, 'r') as file:
    text = file.readlines()


# Extracting training logs
train_lines = [line for line in text if "step=" in line and "Train Loss:" in line]
# Initialize storage for losses grouped by 5000 step intervals
loss_sums = {}
counts = {}

for entry in train_lines:
    # Extracting step and loss values
    step = int(entry.split('step=')[1].split(')')[0])
    loss = float(entry.split('Train Loss: ')[1].split(',')[0])
    
    # Determining the interval
    interval = (step // 5000) * 5000
    
    if interval not in loss_sums:
        loss_sums[interval] = 0
        counts[interval] = 0
    
    loss_sums[interval] += loss
    counts[interval] += 1

# Calculating the average loss for each interval
averages = {interval: loss_sums[interval] / counts[interval] for interval in loss_sums}

for k, avg in averages.items():
    print(str(k) +":" +str(avg))
