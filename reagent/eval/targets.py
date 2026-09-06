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
    # Added to widen the sample: every conclusion drawn from this set rested on
    # ten targets, and several on eight, where one target moving is a visible
    # swing in the reported figures.
    ("acetanilide", "CC(=O)Nc1ccccc1"),
    ("methyl salicylate", "COC(=O)c1ccccc1O"),
    ("caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
    ("theophylline", "Cn1c(=O)c2[nH]cnc2n(C)c1=O"),
    ("vanillin", "COc1cc(C=O)ccc1O"),
    ("diphenhydramine", "CN(C)CCOC(c1ccccc1)c1ccccc1"),
    ("diclofenac", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"),
    ("flurbiprofen", "CC(C(=O)O)c1ccc(-c2ccccc2)c(F)c1"),
    ("mefenamic acid", "Cc1cccc(C)c1Nc1ccccc1C(=O)O"),
    ("phenylbutazone", "CCCCC1C(=O)N(c2ccccc2)N(c2ccccc2)C1=O"),
    ("gabapentin", "NCC1(CC(=O)O)CCCCC1"),
    ("pregabalin", "CC(C)CC(CN)CC(=O)O"),
    ("metformin", "CN(C)C(=N)NC(=N)N"),
    ("atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"),
    ("indomethacin", "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1"),
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
    # Same widening. These run from 16 to 33 heavy atoms, so the set now spans
    # a range of difficulty rather than clustering at one size.
    ("bupropion", "CC(NC(C)(C)C)C(=O)c1cccc(Cl)c1"),
    ("lamotrigine", "Nc1nnc(-c2cccc(Cl)c2Cl)c(N)n1"),
    ("tramadol", "CN(C)CC1CCCCC1(O)c1cccc(OC)c1"),
    ("clopidogrel", "COC(=O)C(c1ccccc1Cl)N1CCc2sccc2C1"),
    ("omeprazole", "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1"),
    ("fluconazole", "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F"),
    ("citalopram", "CN(C)CCCC1(c2ccc(F)cc2)OCc2cc(C#N)ccc21"),
    ("ondansetron", "Cc1nccn1CC1CCc2c(c3ccccc3n2C)C1=O"),
    ("ciprofloxacin", "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"),
    ("donepezil", "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"),
    ("ezetimibe", "O=C1N(c2ccc(F)cc2)C(c2ccc(O)cc2)C1CCC(O)c1ccc(F)cc1"),
    ("losartan", "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1"),
    ("sildenafil", "CCCc1nn(C)c2c1nc([nH]c2=O)-c1cc(S(=O)(=O)N2CCN(C)CC2)ccc1OCC"),
    ("amlodipine", "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl"),
]
