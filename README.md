# REAGENT

Agentic, evidence-grounded framework for automated retrosynthetic planning.

REAGENT plans retrosynthetic routes for a target molecule and evaluates each
candidate route along multiple independent objectives (feasibility, precursor
availability, cost, safety, sustainability, efficiency) using a team of
LLM-backed specialist agents. Chemistry is computed deterministically with
RDKit; the agents interpret those facts, weigh them, and produce a rationale for
the selected route.

The single-step model and tree search are provided by
[AiZynthFinder](https://github.com/MolecularAI/aizynthfinder); REAGENT adds the
agentic evaluation layer on top.

## Requirements

- Python 3.10 or 3.11
- An Anthropic API key (`ANTHROPIC_API_KEY`) for the agent layer

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e .

# Download the pretrained single-step model and building-block stock
download_public_data data
```

## Usage

```powershell
reagent plan "CC(=O)Oc1ccccc1C(=O)O"
```
