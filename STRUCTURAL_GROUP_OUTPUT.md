# Structural Group Output (SGO) — v0.12

## Purpose

Structural Group Output is an interpretation layer for the frozen GSR-FS selected subset. It reports each selected original feature as a structural-group representative and attaches any unselected features that satisfy the frozen Contextual Geometry Substitution (CGS) criterion.

SGO is deliberately **not** a feature-extraction method. It does not average group members, compute latent components, replace columns by group scores, or change the GSR selection path.

## Output semantics

For selected representative `r`, the structural group is

`SG_q = {r} union {j : CGS(r,j | S) >= 0.85}`.

The group identifier `SG01`, `SG02`, ... follows the frozen GSR selection order. The output column of each group is always the selected representative itself.

Every group record contains:

- group ID and GSR selection step;
- representative index/name;
- member indices/names;
- number of contextual substitutes;
- representative GSR selection score;
- effective support, sample coverage, residual novelty and adjusted p-value;
- member-level CGS similarity;
- explicit flags that no latent feature is created and selection is unchanged.

## Python API

```python
from gsrfs import GSRStructuralGroupSelector

selector = GSRStructuralGroupSelector(
    n_features=4,
    substitution_threshold=0.85,
    random_state=42,
)
selector.fit(X)

selector.get_representatives(names=True)
selector.get_structural_groups()
selector.get_structural_group_report()
selector.get_feature_group_map()
selector.get_group_summary()
X_selected = selector.transform_representatives(X)
```

`transform_representatives(X)` is an alias of `transform(X)` and returns only original selected columns.

## CLI

```bash
gsrfs-select data.csv \
  --n-features 4 \
  --structural-groups \
  --substitution-threshold 0.85 \
  --report structural_groups.json
```

The JSON report records `creates_latent_features=false` and `selection_changed_by_grouping=false`.

## Interpretation boundary

A structural group means only that, in the fitted pair-geometry context and relative to the selected GSR basis, the grouped members satisfy the CGS substitution rule. It does **not** imply that group members are causally equivalent, semantically identical, interchangeable in every downstream model, or members of a universal domain factor.

The group output is therefore best treated as an explainability and stability diagnostic for the fitted dataset.
