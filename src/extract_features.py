import csv, statistics
from pathlib import Path
from scapy.all import rdpcap

def extract_features(pcap_path, label):
    try:
        packets = rdpcap(str(pcap_path))
    except Exception as e:
        print(f"Skipping {pcap_path.name}: {e}")
        return None

    times   = [float(p.time) for p in packets if hasattr(p, 'time')]
    lengths = [len(p) for p in packets]

    if len(times) < 2:
        return None

    intervals = [times[i+1] - times[i] for i in range(len(times)-1)]

    return {
        'packet_count':        len(packets),
        'total_length':        sum(lengths),
        'avg_interval':        statistics.mean(intervals),
        'max_interval':        max(intervals),
        'min_interval':        min(intervals),
        'avg_length':          statistics.mean(lengths),
        'max_length':          max(lengths),
        'min_length':          min(lengths),
        'most_common_length':  max(set(lengths), key=lengths.count),
        'label':               label
    }

rows = []
for f in Path("data/raw").glob("*.pcapng"):
    label = f.stem.rsplit('_', 1)[0]
    row = extract_features(f, label)
    if row:
        rows.append(row)
    print(f"Processed {f.name} → {row['label'] if row else 'skipped'}")

with open("data/processed/features.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"\nDone. Saved {len(rows)} records to data/processed/features.csv")