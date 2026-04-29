import csv, statistics
from pathlib import Path
from scapy.all import PcapReader

def extract_features(pcap_path, label):
    times = []
    lengths = []

    try:
        with PcapReader(str(pcap_path)) as packets:
            for p in packets:
                if hasattr(p, "time"):
                    times.append(float(p.time))
                    lengths.append(len(p))
    except Exception as e:
        print(f"\rSkipping {pcap_path.name}: {e}")
        return None

    if len(times) < 2:
        return None

    intervals = [times[i+1] - times[i] for i in range(len(times)-1)]

    return {
        "packet_count": len(lengths),
        "total_length": sum(lengths),
        "avg_interval": statistics.mean(intervals),
        "max_interval": max(intervals),
        "min_interval": min(intervals),
        "avg_length": statistics.mean(lengths),
        "max_length": max(lengths),
        "min_length": min(lengths),
        "most_common_length": max(set(lengths), key=lengths.count),
        "label": label,
    }


directory = list(Path("data/raw").glob("*.pcapng"))
to_process = len(directory)
processed = 0

rows = []
for f in directory:
    print(f"\r[{processed}/{to_process}] Processing {f.name}...".ljust(60), end="", flush=True)

    label = f.stem.rsplit('_', 1)[0]
    row = extract_features(f, label)
    if row:
        rows.append(row)

    processed += 1

with open("data/processed/features.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"\nDone. Saved {len(rows)} records to data/processed/features.csv")