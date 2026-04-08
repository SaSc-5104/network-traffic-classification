import pyshark, os, csv, statistics
from pathlib import Path

def extract_features(pcap_path, label):
    cap = pyshark.FileCapture(str(pcap_path))
    packets = []
    for pkt in cap: 
        try:
            packets.append({
                'time': float(pkt.sniff_timestamp), 
                'length': int(pkt.length)
            })
        except: pass
    cap.close()

    if len(packets) < 2: return None

    times = [p['time'] for p in packets]
    lengths = [p['length'] for p in packets]
    intervals = [times[i+1]-times[i] for i in range(len(times)-1)]

    return {
        'packet_count':       len(packets),
        'total_length':       sum(lengths),
        'avg_interval':      statistics.mean(intervals),
        'max_interval':      max(intervals),
        'min_interval':      min(intervals),
        'avg_length':        statistics.mean(lengths),
        'max_length':        max(lengths),
        'min_length':        min(lengths),
        'most_common_length': max(set(lengths), key=lengths.count),
        'label':             label
    }

rows = []
for f in Path("data/raw").glob("*.pcapng"):
    label = f.stem.rsplit('_',1)[0]
    row = extract_features(f, label)
    if row: rows.append(row)

with open("data/processed/features.csv", "w", newlines = "") as f: 
    w = csv.DictWriter(f, fieldnames = rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"Saved {len(rows)} records")