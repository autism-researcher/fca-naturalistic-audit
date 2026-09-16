# _json_peek.py - prints the structure of every .json in this folder
# Small files are fully parsed; huge files get a safe header peek only.
import json, os, glob, re, sys

def desc(v):
    if isinstance(v, dict):
        return "object with {} keys: {}".format(len(v), list(v.keys())[:6])
    if isinstance(v, list):
        inner = ""
        if v and isinstance(v[0], dict):
            inner = "  (item keys: {})".format(list(v[0].keys())[:8])
        elif v and isinstance(v[0], list):
            inner = "  (item = list of {} values)".format(len(v[0]))
        return "list of {} items{}".format(len(v), inner)
    s = repr(v)
    return s if len(s) < 70 else s[:67] + "..."

files = sorted(glob.glob("*.json"))
if not files:
    print("No .json files found in this folder.")
for f in files:
    size = os.path.getsize(f)
    print("=" * 66)
    print("{}   ({:.1f} MB)".format(f, size / 1e6))
    if size <= 5_000_000:
        try:
            d = json.load(open(f, encoding="utf-8-sig"))
            if isinstance(d, dict):
                for k, v in d.items():
                    print("   {:26} : {}".format(k, desc(v)))
            elif isinstance(d, list):
                print("   top-level: " + desc(d))
            else:
                print("   value: " + desc(d))
        except Exception as e:
            print("   [could not parse: {}]".format(e))
    else:
        print("   LARGE FILE - header peek only (safe, not fully loaded):")
        head = open(f, encoding="utf-8", errors="ignore").read(3000)
        shown = 0
        for m in re.finditer(r'"(\w+)"\s*:\s*("[^"]*"|-?\d+(?:\.\d+)?|\{[^{}]*\})', head):
            print("   {:26} : {}".format(m.group(1), m.group(2)[:70]))
            shown += 1
            if shown >= 8:
                break
        m = re.search(r'"n"\s*:\s*(\d+)', head)
        if m:
            print("   -> contains n = {} trajectories".format(m.group(1)))
        print("   (too big for Notepad; inspect pieces with Python)")
print("=" * 66)
