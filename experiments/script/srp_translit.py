"""Serbian Cyrillic to Latin (Gaj) transliteration and its inverse.

Ran on the GPU cluster; see experiments/README.md.
"""
CYR2LAT = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Ђ": "Đ", "Е": "E",
    "Ж": "Ž", "З": "Z", "И": "I", "Ј": "J", "К": "K", "Л": "L", "Љ": "Lj",
    "М": "M", "Н": "N", "Њ": "Nj", "О": "O", "П": "P", "Р": "R", "С": "S",
    "Т": "T", "Ћ": "Ć", "У": "U", "Ф": "F", "Х": "H", "Ц": "C", "Ч": "Č",
    "Џ": "Dž", "Ш": "Š",
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e",
    "ж": "ž", "з": "z", "и": "i", "ј": "j", "к": "k", "л": "l", "љ": "lj",
    "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "ћ": "ć", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "č",
    "џ": "dž", "ш": "š",
}

LAT2CYR_KEYS = sorted({v for v in CYR2LAT.values()}, key=len, reverse=True)
LAT2CYR = {}
for c, l in CYR2LAT.items():
    LAT2CYR.setdefault(l, c)


def cyr_to_lat(s: str) -> str:
    return "".join(CYR2LAT.get(ch, ch) for ch in s)


def lat_to_cyr(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        for k in LAT2CYR_KEYS:
            if s.startswith(k, i):
                out.append(LAT2CYR[k])
                i += len(k)
                break
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def cyrillic_fraction(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c in CYR2LAT) / len(letters)


if __name__ == "__main__":
    tests = [
        "Његош је написао Горски вијенац.",
        "Џон и Љубица живе у Џакарти, а њихово дијете џогира.",
        "Ђак је дошао у школу с ђубретом.",
        "Надживети некога значи живјети дуже.",
        "Чачак, Шабац и Ужице су градови у Србији.",
    ]
    bad = 0
    for t in tests:
        lat = cyr_to_lat(t)
        back = lat_to_cyr(lat)
        ok = back == t
        bad += (not ok)
        print(f"{'OK ' if ok else 'FAIL'} cyr_frac={cyrillic_fraction(t):.2f}")
        print("   src :", t)
        print("   lat :", lat)
        if not ok:
            print("   back:", back)
    print(f"\n{len(tests) - bad}/{len(tests)} round trips exact")
