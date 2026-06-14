"""A small fixed set of drug-like target molecules for evaluation."""

# (name, SMILES). Common, well-known molecules spanning easy to moderate routes.
TARGETS: list[tuple[str, str]] = [
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
    ("benzocaine", "CCOC(=O)c1ccc(N)cc1"),
    ("phenacetin", "CCOc1ccc(NC(C)=O)cc1"),
    ("salicylamide", "NC(=O)c1ccccc1O"),
    ("procaine", "CCN(CC)CCOC(=O)c1ccc(N)cc1"),
    ("lidocaine", "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
    ("naproxen", "COc1ccc2cc(C(C)C(=O)O)ccc2c1"),
    ("ketoprofen", "CC(C(=O)O)c1cccc(C(=O)c2ccccc2)c1"),
]

# Larger, more complex drugs whose routes need several steps. These stress the
# planner and the multi-step feature/agent logic; most need a permissive stock
# to route at all.
HARD_TARGETS: list[tuple[str, str]] = [
    ("fluoxetine", "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1"),
    ("sertraline", "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21"),
    ("diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"),
    ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
    ("celecoxib", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"),
    ("sumatriptan", "CNS(=O)(=O)Cc1ccc2[nH]cc(CCN(C)C)c2c1"),
    ("chlorpromazine", "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21"),
    ("venlafaxine", "COc1ccc(C(CN(C)C)C2(O)CCCCC2)cc1"),
    ("propranolol", "CC(C)NCC(O)COc1cccc2ccccc12"),
    ("ketamine", "CNC1(c2ccccc2Cl)CCCCC1=O"),
]
