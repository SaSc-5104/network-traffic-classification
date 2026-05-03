import csv
import logging
import numpy as np
from pathlib import Path
from scapy.all import TCP, UDP, PcapReader, load_layer

logging.getLogger('scapy.runtime').setLevel(logging.ERROR)

try:
    load_layer('tls')
except Exception:
    pass


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0


def has_tls_layer(packet):
    try:
        return packet.haslayer('TLS') or packet.haslayer('TLS Record')
    except Exception:
        return False

def extract_features(pcap_path, label):
    times = []
    lengths = []
    tcp_packets = 0
    udp_packets = 0
    tls_packets = 0
    quic_packets = 0
    port_443_packets = 0
    destination_ports = set()

    try:
        with PcapReader(str(pcap_path)) as packets:
            for p in packets:
                if hasattr(p, 'time'):
                    times.append(float(p.time))
                    lengths.append(len(p))

                    tcp_layer = p.getlayer(TCP)
                    udp_layer = p.getlayer(UDP)

                    if tcp_layer:
                        tcp_packets += 1
                        destination_ports.add(int(tcp_layer.dport))
                        if tcp_layer.sport == 443 or tcp_layer.dport == 443:
                            port_443_packets += 1

                    if udp_layer:
                        udp_packets += 1
                        destination_ports.add(int(udp_layer.dport))
                        if udp_layer.sport == 443 or udp_layer.dport == 443:
                            port_443_packets += 1
                            quic_packets += 1

                    if has_tls_layer(p):
                        tls_packets += 1
    except Exception as e:
        print(f'\rSkipping {pcap_path.name}: {e}')
        return None

    if len(times) < 2:
        return None

    intervals = np.array([times[i+1] - times[i] for i in range(len(times)-1)])
    lengths = np.array(lengths)
    duration = max(times) - min(times)
    duration = duration if duration > 0 else 1e-9
    packet_total = len(lengths)

    return {
        'interval_mean': intervals.mean(),
        'interval_stdev': intervals.std(),
        'interval_min': intervals.min(),
        'interval_max': intervals.max(),
        'interval_p25': np.percentile(intervals, 25),
        'interval_median': np.percentile(intervals, 50),
        'interval_p75': np.percentile(intervals, 75),
        'interval_p90': np.percentile(intervals, 90),
        'length_mean': lengths.mean(),
        'length_stdev': lengths.std(),
        'length_min': lengths.min(),
        'length_max': lengths.max(),
        'length_p25': np.percentile(lengths, 25),
        'length_median': np.percentile(lengths, 50),
        'length_p75': np.percentile(lengths, 75),
        'length_p90': np.percentile(lengths, 90),
        'packets_per_second': len(times) / duration,
        'bytes_per_second': lengths.sum() / duration,
        'tcp_packets_per_second': tcp_packets / duration,
        'udp_packets_per_second': udp_packets / duration,
        'tls_packets_per_second': tls_packets / duration,
        'quic_packets_per_second': quic_packets / duration,
        'tcp_packet_ratio': safe_divide(tcp_packets, packet_total),
        'udp_packet_ratio': safe_divide(udp_packets, packet_total),
        'tls_packet_ratio': safe_divide(tls_packets, packet_total),
        'tls_or_quic_packet_ratio': safe_divide(tls_packets + quic_packets, packet_total),
        'quic_udp_443_ratio': safe_divide(quic_packets, udp_packets),
        'most_common_length': int(np.bincount(lengths).argmax()),
        'label': label,
    }


directory =  [
    p for p in Path('data/raw').glob('*.pcapng')
    if not (p.name.startswith('handshake') 
        or p.name.startswith('linkedin') 
        or p.name.startswith('wp')
        or p.name.startswith('tetris'))
]
to_process = len(directory)
processed = 0

rows = []
for f in directory:
    print(f'\r[{processed}/{to_process}] Processing {f.name}...'.ljust(60), end='', flush=True)

    label = f.stem.rsplit('_', 1)[0]
    row = extract_features(f, label)
    if row:
        rows.append(row)

    processed += 1

with open('data/processed/features.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f'\nDone. Saved {len(rows)} records to data/processed/features.csv')
