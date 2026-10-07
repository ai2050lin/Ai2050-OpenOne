import os, json

fe = r"D:\AI2050\Ai2050-OpenOne\frontend"
nm = os.path.join(fe, "node_modules")
report = {"node_modules_exists": os.path.isdir(nm)}
if report["node_modules_exists"]:
    entries = sorted(os.listdir(nm))
    report["top_level_count"] = len(entries)
    report["has_vite"] = os.path.isdir(os.path.join(nm, "vite"))
    report["has_vitejs"] = os.path.isdir(os.path.join(nm, "@vitejs"))
    report["has_react"] = os.path.isdir(os.path.join(nm, "react"))
    report["has_three"] = os.path.isdir(os.path.join(nm, "three"))
    report["has_plotly"] = os.path.isdir(os.path.join(nm, "plotly.js"))
    report["has_dot_package_lock"] = os.path.exists(os.path.join(nm, ".package-lock.json"))
    report["scoped"] = [e for e in entries if e.startswith("@")]
report["pkg_lock_exists"] = os.path.exists(os.path.join(fe, "package-lock.json"))
report["vite_config_exists"] = any(f.startswith("vite.config") for f in os.listdir(fe))
report["frontend_top"] = sorted(os.listdir(fe))

node_root = r"C:\Users\Admin\.workbuddy\binaries\node\versions"
report["managed_node_versions"] = sorted(os.listdir(node_root))
v = os.path.join(node_root, "22.22.2-3")
report["v2222_files"] = sorted(os.listdir(v)) if os.path.isdir(v) else None

out = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_probe_frontend.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write(json.dumps(report, ensure_ascii=False, indent=1))
print("WROTE", out)
