"""Pull the v630 neuron ID sets (sugar, bitter, water, JO, MN9...) straight out of the paper's figures notebook."""
import json, re
nb = json.load(open("shiu/figures.ipynb"))
src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
sets = {}
for m in re.finditer(r"(\w+)\s*=\s*\[([^\]]*?7205[^\]]*)\]", src, re.S):
    sets.setdefault(m.group(1), [int(x) for x in re.findall(r"\d{18}", m.group(2))])
json.dump(sets, open("sets630.json", "w")); print({k: len(v) for k, v in sets.items()})
