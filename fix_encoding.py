f = r"c:\Users\USER\Desktop\FL Security\TAWOS_FL_FEASIBILITY.html"

with open(f, 'r', encoding='utf-8-sig') as fh:   # utf-8-sig strips BOM
    c = fh.read()

def fix_mojibake(text):
    """
    Reverses the double-encoding: characters were originally UTF-8 bytes
    that got misread as Latin-1/Windows-1252 and re-saved as UTF-8.
    Fix: encode non-ASCII chars back to Latin-1 bytes, then re-decode as UTF-8.
    """
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ord(ch) > 127:
            # Collect a run of non-ASCII chars
            seq = []
            j = i
            while j < len(text) and ord(text[j]) > 127:
                # Handle Windows-1252 special range 0x80-0x9F
                cp = ord(text[j])
                w1252_map = {
                    0x20AC: 0x80, 0x201A: 0x82, 0x0192: 0x83, 0x201E: 0x84,
                    0x2026: 0x85, 0x2020: 0x86, 0x2021: 0x87, 0x02C6: 0x88,
                    0x2030: 0x89, 0x0160: 0x8A, 0x2039: 0x8B, 0x0152: 0x8C,
                    0x017D: 0x8E, 0x2018: 0x91, 0x2019: 0x92, 0x201C: 0x93,
                    0x201D: 0x94, 0x2022: 0x95, 0x2013: 0x96, 0x2014: 0x97,
                    0x02DC: 0x98, 0x2122: 0x99, 0x0161: 0x9A, 0x203A: 0x9B,
                    0x0153: 0x9C, 0x017E: 0x9E, 0x0178: 0x9F,
                }
                byte = w1252_map.get(cp, cp if cp <= 0xFF else None)
                if byte is None:
                    seq = None
                    break
                seq.append(byte)
                j += 1
            if seq is not None:
                try:
                    fixed = bytes(seq).decode('utf-8')
                    result.append(fixed)
                    i = j
                    continue
                except (UnicodeDecodeError, ValueError):
                    pass
        result.append(ch)
        i += 1
    return ''.join(result)

fixed = fix_mojibake(c)

with open(f, 'w', encoding='utf-8') as fh:
    fh.write(fixed)

# Report
lines = fixed.split('\n')
bad = [l for l in lines if any(ord(ch) > 127 for ch in l)]
print(f"Done. Lines still containing non-ASCII: {len(bad)}")
for l in bad[:5]:
    print(repr(l.strip()[:120]))
