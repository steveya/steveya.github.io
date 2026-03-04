"""Apply prose consistency fixes to draft.ipynb"""

import json

with open("draft.ipynb") as f:
    nb = json.load(f)

fixes_applied = []

for cell in nb["cells"]:
    if cell["cell_type"] != "markdown":
        continue
    src = "".join(cell["source"])

    # Fix: Procrustes drift 1.3x -> 5-7x in cell 26
    old = "about 1.3$\\times$ the drift of PCA/Tenor on average"
    new = "roughly 5--7$\\times$ the drift of PCA/Tenor on average"
    if old in src:
        src = src.replace(old, new)
        lines = src.split("\n")
        cell["source"] = [l + "\n" for l in lines[:-1]] + (
            [lines[-1]] if lines[-1] else []
        )
        fixes_applied.append("Procrustes drift 1.3x -> 5-7x")

    # Fix: Score roughness + loading roughness in Discussion
    old = (
        "Empirical score roughness $\\operatorname{tr}(U^\\top \\widetilde{L}_s U)$ "
        "ranges from 600 to 3{,}500 across windows---large day-to-day jumps---"
        "while loading roughness $\\operatorname{tr}(V^\\top \\widetilde{L}_f V) "
        "\\approx 0.04$ places PCA loadings near the null space of the tenor-chain Laplacian."
    )
    new = (
        "Empirical score roughness $\\operatorname{tr}(U^\\top \\widetilde{L}_s U)$ "
        "averages roughly 400 in one-year windows and 850 in three-month windows---"
        "large day-to-day jumps---while loading roughness "
        "$\\operatorname{tr}(V^\\top \\widetilde{L}_f V)$ is approximately 0.01--0.04 "
        "across settings, placing PCA loadings near the null space of the tenor-chain Laplacian."
    )
    if old in src:
        src = src.replace(old, new)
        lines = src.split("\n")
        cell["source"] = [l + "\n" for l in lines[:-1]] + (
            [lines[-1]] if lines[-1] else []
        )
        fixes_applied.append(
            "Score roughness 600-3500 -> 400/850; loading roughness 0.04 -> 0.01-0.04"
        )

for fix in fixes_applied:
    print(f"  Applied: {fix}")

if not fixes_applied:
    print("  WARNING: No fixes matched!")
else:
    with open("draft.ipynb", "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"Saved {len(fixes_applied)} fixes.")
