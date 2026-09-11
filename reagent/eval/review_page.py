"""Standalone, offline literature-review worksheet with RDKit molecular drawings."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from reagent.eval.reference_review import digest, report


def render_page(directory: Path) -> Path:
    bundle, reviews, references = [
        json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("bundle", "reviews", "references")
    ]
    report(bundle, reviews, references)  # refuse stale or incomplete worksheets
    literature_file = directory / "literature.json"
    literature = json.loads(literature_file.read_text()) if literature_file.exists() else {
        "bundle_sha256": digest(bundle), "targets": {},
    }
    if literature.get("bundle_sha256") != digest(bundle):
        raise ValueError("Literature notes do not belong to this bundle")
    smiles = set()
    for row in bundle["targets"]:
        smiles.add(row["target"])
        if row["route"]:
            for reaction in row["route"]["reactions"]:
                smiles.update([reaction["product"], *reaction["precursors"]])
    drawings = {}
    for value in sorted(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"Cannot draw invalid SMILES: {value!r}")
        drawer = rdMolDraw2D.MolDraw2DSVG(320, 200)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        drawings[value] = "data:image/svg+xml;base64," + base64.b64encode(
            drawer.GetDrawingText().encode()
        ).decode()
    payload = json.dumps({
        "targets": bundle["targets"], "reviews": reviews,
        "digest": digest(bundle), "drawings": drawings, "literature": literature["targets"],
    }, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    path = directory / "review.html"
    html = _PAGE.replace("__TARGET_COUNT__", str(len(bundle["targets"])))
    path.write_text(html.replace("__PAYLOAD__", payload), encoding="utf-8")
    return path


_PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ReAgent · Check the evidence</title>
<style>
body{font:16px/1.55 system-ui,sans-serif;background:#f4f6f8;color:#172536;margin:0}
header,main{max-width:1100px;margin:auto;padding:24px}header{padding-bottom:0}
h1{margin:0}h2{margin-top:0}a{color:#1253a3}button,select,input,textarea{font:inherit}
button,select,input{padding:9px;border:1px solid #9aaaba;border-radius:6px;background:white}
button{cursor:pointer}button:disabled{opacity:.45}textarea{width:100%;box-sizing:border-box;min-height:95px}
nav,.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:16px 0}
.card{background:white;border:1px solid #d5dee7;border-radius:10px;padding:22px;margin:18px 0}
.scheme{display:flex;align-items:center;flex-wrap:wrap;gap:10px}.molecule{width:230px;margin:0}
.molecule img{width:100%;height:auto}.molecule code{display:block;font-size:11px;overflow-wrap:anywhere}
.molecule a{font-size:13px}.arrow{font-size:28px}.muted{color:#526373}.note{border-left:4px solid #c48c2c;padding-left:15px}
label{display:block;margin-top:12px}#message{min-height:24px}summary{cursor:pointer}
@media print{nav,.toolbar,button,#message{display:none}.card{break-inside:avoid}}
</style>
<header><p class="muted">ReAgent · Literature review</p><h1>Check the evidence yourself</h1>
<p>__TARGET_COUNT__ targets, one saved selection per solved target. Work through each reaction, open its sources,
and record what you can substantiate. This page never contacts anyone or calls an AI model.</p>
<details class="card"><summary>What counts as confirmation?</summary>
<ol><li>Check the exact starting materials and product, including stereochemistry.</li>
<li>Find the transformation in a primary paper or patent. Record its DOI/URL and scheme or example.</li>
<li>Check reagents, catalysts, conditions, selectivity and reported outcome. The drawings below omit
conditions; a similar transformation or a search result alone does not establish support.</li>
<li>Use <b>Supported</b> only for the claim supported by the source and explain its limits.
Use <b>Unsupported</b> for a documented contradiction, and <b>Uncertain</b> when evidence is missing
or ambiguous. Failure to find a paper is not proof a reaction is impossible.</li>
<li>Review the full route separately. Matching individual steps does not establish that the whole
sequence has been demonstrated.</li></ol>
<p>This is your literature check, not an independent expert review or a laboratory validation.
Many targets omit stereochemistry. No percentage here establishes experimental accuracy.</p></details>
<div class="toolbar"><label>Your name or initials <input id="reviewer" placeholder="Recorded on your judgments"></label>
<button id="download">Export reviews.json</button>
<label>Load saved reviews <input id="upload" type="file" accept=".json,application/json"></label></div>
<p class="muted">Drafts are saved in this browser when storage is available. Export a file for a durable copy.
The page cannot overwrite your original worksheets. Imported reviews replace the current draft.</p>
<p id="message" role="status" aria-live="polite"></p>
<nav><button id="previous">Previous</button><select id="target" aria-label="Target"></select>
<button id="next">Next</button><span id="progress"></span></nav></header>
<main id="content"></main>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent);
const storageKey='reagent-review-'+data.digest;
let reviews=structuredClone(data.reviews), position=0;
const $=id=>document.getElementById(id);
const verdicts=['unreviewed','supported','unsupported','uncertain'];
const esc=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function validate(value,complete=true){
 if(value.schema_version!==1||value.bundle_sha256!==data.digest)throw Error('Different review bundle.');
 if(!['unspecified','self_literature','expert'].includes(value.review_basis||'unspecified'))throw Error('Invalid review basis.');
 const ids=Object.keys(data.reviews.targets).sort();
 if(!value.targets||JSON.stringify(Object.keys(value.targets).sort())!==JSON.stringify(ids))throw Error('Target list differs.');
 for(const id of ids){const r=value.targets[id];
  if(!Array.isArray(r.steps)||r.steps.length!==data.reviews.targets[id].steps.length)throw Error('Step count differs.');
  for(const j of [r.route,...r.steps]){
   if(!j||!verdicts.includes(j.verdict)||typeof j.reviewer!=='string'||typeof j.evidence!=='string')throw Error('Invalid judgment.');
   if(complete&&j.verdict!=='unreviewed'&&(!j.reviewer.trim()||!j.evidence.trim()))throw Error('Each judgment needs your name and evidence.');
  }
 }
}
function message(text){$('message').textContent=text;}
try{const cached=localStorage.getItem(storageKey);if(cached){const parsed=JSON.parse(cached);validate(parsed,false);reviews=parsed;message('Restored saved browser draft.');}}
catch(e){message('Browser draft unavailable or incomplete. Use exported files to preserve work.');}
function save(){reviews.review_basis='self_literature';try{localStorage.setItem(storageKey,JSON.stringify(reviews));message('Draft saved in this browser. Export for a durable copy.');}catch(e){message('Browser storage unavailable. Export your work before closing.');}progress();}
function progress(){const rows=Object.values(reviews.targets);let completed=0,total=0;
 for(const r of rows)for(const j of [r.route,...r.steps]){total++;if(j.verdict!=='unreviewed')completed++;}
 $('progress').textContent=completed+' / '+total+' judgments recorded';}
function molecule(smiles){return '<figure class="molecule"><img alt="Molecular structure: '+esc(smiles)+'" src="'+data.drawings[smiles]+'"><code>'+esc(smiles)+'</code><a target="_blank" rel="noopener noreferrer" href="https://pubchem.ncbi.nlm.nih.gov/#query='+encodeURIComponent(smiles)+'">Look up molecule identity</a></figure>';}
function form(id,index,title){const judgment=index<0?reviews.targets[id].route:reviews.targets[id].steps[index];
 return '<section class="card"><h3>'+title+'</h3><label>Conclusion <select data-index="'+index+'" data-field="verdict">'+verdicts.map(v=>'<option value="'+v+'"'+(v===judgment.verdict?' selected':'')+'>'+v[0].toUpperCase()+v.slice(1)+'</option>').join('')+'</select></label>'+
 '<label>Evidence and limits <textarea data-index="'+index+'" data-field="evidence" placeholder="DOI or URL; scheme/example; exact match or differences; missing conditions and stereochemistry">'+esc(judgment.evidence)+'</textarea></label><p class="muted">Recorded by: <span data-author="'+index+'">'+esc(judgment.reviewer||'not yet recorded')+'</span></p></section>';}
function render(){const row=data.targets[position];$('target').value=String(position);$('previous').disabled=position===0;$('next').disabled=position===data.targets.length-1;
 let html='<section class="card"><h2>'+esc(row.name)+' <small class="muted">'+(position+1)+' / '+data.targets.length+'</small></h2>'+molecule(row.target)+
 '<p><a target="_blank" rel="noopener noreferrer" href="https://scholar.google.com/scholar?q='+encodeURIComponent(row.name+' synthesis')+'">Search papers</a> · '+
 '<a target="_blank" rel="noopener noreferrer" href="https://patents.google.com/?q='+encodeURIComponent(row.name+' synthesis')+'">Search patents</a></p><p class="muted">Search links are research leads, not supporting evidence.</p></section>';
 if(!row.route){html+='<section class="card">No solved route was saved for this target. It remains in the full cohort denominator.</section>';}
 else{row.route.reactions.forEach((reaction,i)=>{
  html+='<section class="card"><h3>Saved step '+(i+1)+'</h3><p class="muted">Shown in synthesis direction. Step numbers follow the saved retrosynthetic list, not execution order.</p><div class="scheme">'+reaction.precursors.map(molecule).join('<span class="arrow">+</span>')+'<span class="arrow">→</span>'+molecule(reaction.product)+'</div><p class="note">Reagents, conditions and yields are not specified in this saved reaction.</p>';
  for(const source of data.literature[row.id]||[]){if(source.step!==i+1)continue;
   let url;try{url=new URL(source.url);}catch(e){continue;}if(!['https:','http:'].includes(url.protocol))continue;
   html+='<p><a target="_blank" rel="noopener noreferrer" href="'+esc(url.href)+'">'+esc(source.title)+'</a></p><p>'+esc(source.note)+'</p><p class="muted">Source location: '+esc(source.locator)+' · Research lead; no verdict assigned.</p>';
  }
  html+='</section>'+form(row.id,i,'Your check of step '+(i+1));
 });html+=form(row.id,-1,'Your check of the complete route');}
 $('content').innerHTML=html;
 $('content').querySelectorAll('[data-field]').forEach(element=>element.addEventListener('input',()=>{
  const index=Number(element.dataset.index),j=index<0?reviews.targets[row.id].route:reviews.targets[row.id].steps[index];
  j[element.dataset.field]=element.value;j.reviewer=$('reviewer').value.trim()||j.reviewer;
  $('content').querySelector('[data-author="'+index+'"]').textContent=j.reviewer||'not yet recorded';save();
 }));progress();}
data.targets.forEach((row,i)=>{const option=document.createElement('option');option.value=i;option.textContent=(i+1)+'. '+row.name+(row.route?'':' (unsolved)');$('target').append(option);});
$('target').addEventListener('change',()=>{position=Number($('target').value);render();});
$('previous').onclick=()=>{position--;render();};$('next').onclick=()=>{position++;render();};
$('download').onclick=()=>{try{reviews.review_basis='self_literature';validate(reviews);
 const url=URL.createObjectURL(new Blob([JSON.stringify(reviews,null,2)+'\n'],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='reviews.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);message('Exported reviews.json. Keep it with the matching bundle.');
 }catch(e){message('Cannot export: '+e.message+' Complete the name/evidence fields or leave the verdict unreviewed.');}};
$('upload').onchange=async event=>{try{const file=event.target.files[0];if(!file)return;const parsed=JSON.parse(await file.text());validate(parsed);reviews=parsed;save();render();message('Imported reviews. Export again after making changes.');}catch(e){message('Import rejected: '+e.message);}};
render();
</script></html>'''
