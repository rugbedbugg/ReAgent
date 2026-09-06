"""The evaluation sets are data, and wrong data is worse than no data.

A target whose SMILES parses but describes the wrong molecule corrupts every
figure computed from it, silently and permanently, because nothing downstream
re-checks it. One candidate for this set (rosuvastatin) was written with its
fluorophenyl group missing: it parsed cleanly, RDKit was happy, and only the
molecular formula gave it away. Hence the formula check rather than a bare
parse check.
"""

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from reagent.eval.targets import HARD_TARGETS, TARGETS

RDLogger.DisableLog("rdApp.*")

# Reference molecular formulas, independent of the SMILES they are checking.
FORMULAS = {
    "aspirin": "C9H8O4", "paracetamol": "C8H9NO2", "ibuprofen": "C13H18O2",
    "benzocaine": "C9H11NO2", "phenacetin": "C10H13NO2", "salicylamide": "C7H7NO2",
    "procaine": "C13H20N2O2", "lidocaine": "C14H22N2O", "naproxen": "C14H14O3",
    "ketoprofen": "C16H14O3", "acetanilide": "C8H9NO",
    "methyl salicylate": "C8H8O3", "caffeine": "C8H10N4O2",
    "theophylline": "C7H8N4O2", "vanillin": "C8H8O3",
    "diphenhydramine": "C17H21NO", "diclofenac": "C14H11Cl2NO2",
    "flurbiprofen": "C15H13FO2", "mefenamic acid": "C15H15NO2",
    "phenylbutazone": "C19H20N2O2", "gabapentin": "C9H17NO2",
    "pregabalin": "C8H17NO2", "metformin": "C4H11N5", "atenolol": "C14H22N2O3",
    "indomethacin": "C19H16ClNO4",
    "fluoxetine": "C17H18F3NO", "sertraline": "C17H17Cl2N", "diazepam": "C16H13ClN2O",
    "warfarin": "C19H16O4", "celecoxib": "C17H14F3N3O2S", "sumatriptan": "C14H21N3O2S",
    "chlorpromazine": "C17H19ClN2S", "venlafaxine": "C17H27NO2",
    "propranolol": "C16H21NO2", "ketamine": "C13H16ClNO",
    "bupropion": "C13H18ClNO", "lamotrigine": "C9H7Cl2N5", "tramadol": "C16H25NO2",
    "clopidogrel": "C16H16ClNO2S", "omeprazole": "C17H19N3O3S",
    "fluconazole": "C13H12F2N6O", "citalopram": "C20H21FN2O",
    "ondansetron": "C18H19N3O", "ciprofloxacin": "C17H18FN3O3",
    "donepezil": "C24H29NO3", "ezetimibe": "C24H21F2NO3", "losartan": "C22H23ClN6O",
    "sildenafil": "C22H30N6O4S", "amlodipine": "C20H25ClN2O5",
}

ALL_TARGETS = TARGETS + HARD_TARGETS


def test_every_target_parses():
    for name, smiles in ALL_TARGETS:
        assert Chem.MolFromSmiles(smiles) is not None, f"{name}: unparseable SMILES"


def test_every_target_matches_its_reference_formula():
    """The check that catches a plausible-but-wrong structure."""
    for name, smiles in ALL_TARGETS:
        assert name in FORMULAS, f"{name}: no reference formula, add one"
        molecule = Chem.MolFromSmiles(smiles)
        formula = rdMolDescriptors.CalcMolFormula(molecule).rstrip("+-")
        assert formula == FORMULAS[name], f"{name}: {formula} != {FORMULAS[name]}"


def test_target_names_are_unique_within_and_across_sets():
    names = [n for n, _ in ALL_TARGETS]
    assert len(names) == len(set(names))


def test_the_sets_are_large_enough_to_report_on():
    """Conclusions were previously drawn from ten targets, where one target
    moving is a visible swing in the reported figures."""
    assert len(TARGETS) >= 25
    assert len(HARD_TARGETS) >= 24


def test_hard_targets_are_actually_harder():
    """The hard set should sit above the moderate one in size, or it is not
    measuring anything different."""
    def mean_heavy_atoms(targets):
        counts = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for _, s in targets]
        return sum(counts) / len(counts)

    assert mean_heavy_atoms(HARD_TARGETS) > mean_heavy_atoms(TARGETS)
