# Menguji sel Bab 8 pada CSV tiruan berstruktur mirip (HANYA untuk menguji kode; hasilnya tidak dipakai di buku)
import os, sys, io, contextlib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
work = sys.argv[1]; os.makedirs(work + "/data", exist_ok=True); os.chdir(work)
rng = np.random.default_rng(0)
cnt = {"DOS_SYN_Hping":9465,"ARP_poisioning":775,"NMAP_UDP_SCAN":259,"NMAP_XMAS_TREE_SCAN":201,"NMAP_OS_DETECTION":200,
       "NMAP_TCP_scan":100,"DDOS_Slowloris":53,"Metasploit_Brute_Force_SSH":12,"NMAP_FIN_SCAN":9,"MQTT":810,"Thing_speak":414,
       "Wipro_bulb_Dataset":25,"Amazon-Alexa":8684}
rows = []
for i,(k,n) in enumerate(cnt.items()):
    mu = rng.normal(0,1.2,6)
    num = rng.normal(mu, 1.0, (n,6)).round(3)
    proto = rng.choice(["tcp","udp","icmp"], n, p=[.6,.3,.1])
    service = rng.choice(["-","mqtt","http","dns","ssh"], n)
    rows.append(pd.DataFrame(num, columns=[f"f{j}" for j in range(6)]).assign(proto=proto, service=service, Attack_type=k))
d = pd.concat(rows).sample(frac=1, random_state=1).reset_index(drop=True)
d["f5"] = (d["f5"] * 3).round(0)                       # nilai bulat -> memicu duplikat
dup = d.sample(600, random_state=2); d = pd.concat([d, dup]).reset_index(drop=True)
d.loc[d.sample(15, random_state=3).index, "f1"] = np.nan
d.insert(0, "Unnamed: 0", range(len(d)))
d.to_csv("data/RT_IOT2022.csv", index=False)
ns = {}
for f in sys.argv[2:]:
    src = "\n".join(l for l in open(f).read().splitlines() if not l.startswith("!"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): exec(compile(src, f, "exec"), ns)
    print("=====", os.path.basename(f)); print(buf.getvalue()[:1800])
