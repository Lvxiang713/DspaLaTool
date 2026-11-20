"""Standalone English report and dependency-free quadratic predictor."""
import html
import json
from pathlib import Path


def rendered_equation(artifact, encoder):
    """Native MathML, offline and without external scripts or fonts."""
    sub=lambda a,b:f"<msub><mi>{a}</mi><mi>{b}</mi></msub>"
    z,x,mu,sigma=[sub(a,"j") for a in ("z","x","μ","σ")]
    normalization=f"<math display='block'><mrow>{z}<mo>=</mo><mfrac><mrow>{x}<mo>−</mo>{mu}</mrow>{sigma}</mfrac></mrow></math>"
    if artifact["kind"]=="native_taylor":
        order=artifact["approximation_order"]
        expression="<msub><mi>b</mi><mi>c</mi></msub><mo>+</mo><msubsup><mi>g</mi><mi>c</mi><mi>T</mi></msubsup><mi>u</mi>"
        if order>=2:
            expression+="<mo>+</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><msup><mi>u</mi><mi>T</mi></msup><msub><mi>H</mi><mi>c</mi></msub><mi>u</mi>"
        if order==3:
            expression+="<mo>+</mo><mfrac><mn>1</mn><mn>6</mn></mfrac><munder><mo>∑</mo><mrow><mi>i</mi><mi>j</mi><mi>k</mi></mrow></munder><msub><mi>T</mi><mrow><mi>c</mi><mi>i</mi><mi>j</mi><mi>k</mi></mrow></msub>"+sub("u","i")+sub("u","j")+sub("u","k")
        expression=expression.replace("<mi>u</mi>","<mi>z</mi>")
        note=""
    else:
        expression=(sub("β","c")+"<mo>+</mo><munder><mo>∑</mo><mi>j</mi></munder><mo>(</mo>"+sub("A","cj")+f"<msup>{z}<mn>2</mn></msup><mo>+</mo>"+sub("B","cj")+z+"<mo>)</mo>")
        note=""
    from .equation_view import equation_preview
    core="<math display='block'>"+sub("S","c")+"<mo>=</mo>"+expression+"</math>"
    view=equation_preview(artifact,encoder,core)
    return view.replace('<div class="full-equation"',normalization+note+'<div class="full-equation"',1)

def decision_equation(classes):
    if len(classes or []) != 2:
        return ""
    return ("<math display='block'><mi>Q</mi><mo>=</mo><msub><mi>S</mi><mn>1</mn></msub>"
            "<mo>−</mo><msub><mi>S</mi><mn>0</mn></msub></math>")


def write_report(output, summary):
    # Score RMSE/R2 are intentionally omitted from user-facing results.
    summary=json.loads(json.dumps(summary))
    for row in summary.get("results",{}).values():
        row.pop("score_RMSE",None)
        row.pop("score_R2",None)
    is_taylor = summary.get("equation_kind") == "native_taylor"
    expression_label = "Taylor expression" if is_taylor else "Quadratic"
    delta_label = "Δ expression − original" if is_taylor else "Δ quadratic − original"
    text = html.escape(json.dumps(summary, indent=2, allow_nan=False))
    rows = []
    fidelity_rows = []
    monitoring_section = ""
    history_path = output / "monitor.jsonl"
    if history_path.exists():
        history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        from .monitoring import display_history
        history=display_history(history,summary.get("training_n"))
        history = [{k:v for k,v in row.items() if k in
                    ("epoch","total_loss_mean")} for row in history]
        helper = Path(__file__).with_name("loss_view.js").read_text(encoding="utf-8")
        payload = json.dumps(history, allow_nan=False).replace("<","\\u003c")
        monitoring_section = (
            '<h2>Training losses</h2>'
            '<h3>Mean Total Loss</h3><canvas id="totalChart" width="1000" height="210" style="width:100%"></canvas>'
            '<script>'+helper+'</script><script>const history='+payload+';'
            'TrainingView.loss(document.querySelector("#totalChart"),history,"total_loss_mean","Mean Total Loss","#173248");'
            '</script>')
    def fmt(value):
        return "Undefined" if value is None else f"{value:.4f}"
    original_shap_image = ('<h2>Original-model explanation</h2><p>Sampled grouped permutation SHAP, '
                           'separate from the extracted-expression SHAP analysis.</p>'
                           '<img src="original_model_SHAP.png" alt="Original model sampled SHAP ranking">'
                           if summary.get("original_model_SHAP") else "")
    for dataset, result in summary["results"].items():
        for metric in dict.fromkeys([*(result["original"] or {}), *(result["quadratic"] or {})]):
            original, quadratic = result["original"].get(metric), result["quadratic"].get(metric)
            delta = None if original is None or quadratic is None else quadratic-original
            rows.append(f"<tr><td>{html.escape(dataset)}</td><td>{html.escape(metric)}</td>"
                        f"<td>{fmt(original)}</td><td>{fmt(quadratic)}</td><td>{fmt(delta)}</td></tr>")
        for metric in ("prediction_agreement",):
            if metric in result:
                fidelity_rows.append(f"<tr><td>{html.escape(dataset)}</td><td>{metric}</td>"
                                     f"<td>{fmt(result[metric])}</td></tr>")
    artifact = json.loads((output / "equation.json").read_text(encoding="utf-8"))
    encoder = json.loads((output / "encoder.json").read_text(encoding="utf-8"))
    coefficient_rows = []
    for c, equation in enumerate(artifact["classes"]):
        label = str(summary["classes"][c]) if summary["classes"] else "Regression"
        coefficient_rows.append(f"<tr><td>{html.escape(label)}</td><td>Intercept</td>"
                                f"<td colspan='4'>{equation['intercept']:.8g}</td></tr>")
        for j, name in enumerate(encoder["names"]):
            coefficient_rows.append("<tr><td>"+html.escape(label)+"</td><td>"+html.escape(name)+"</td>"
                                    +"".join(f"<td>{value:.8g}</td>" for value in
                                             (artifact["mean"][j], artifact["std"][j],
                                              equation["A"][j], equation["B"][j]))+"</tr>")
    equation_section = (
        '<h2>The explicit prediction equation</h2>'
        + rendered_equation(artifact,encoder) +
        decision_equation(summary["classes"])+
        '<details><summary>Show all equation coefficients and normalization statistics</summary>'
        '<table><tr><th>Class</th><th>Encoded feature</th><th>μ</th><th>σ</th><th>A</th><th>B</th></tr>'
        +"".join(coefficient_rows)+'</table></details>')
    if artifact["kind"] == "native_taylor":
        order = artifact["approximation_order"]
        equation_section = (
            '<h2>The explicit Taylor prediction equation</h2>'
            + rendered_equation(artifact,encoder) + decision_equation(summary["classes"]) +
            '<p><a href="coefficients.csv">Main coefficients</a> · '
            '<a href="interaction_coefficients.csv">Second-order interaction coefficients</a>'
            + (' · <a href="cubic_coefficients.csv">Third-order coefficients</a>' if order==3 else '')+'</p>')
    (output / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Tabular Agent analysis</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#183047}'
        'math{font-size:1.25em;margin:18px 0}.equation-math{overflow-x:auto;background:#f3f8fb;padding:18px;border-radius:10px}'
        'pre{white-space:pre-wrap;background:#f0f5f8;padding:24px}img{max-width:100%}'
        'table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #dde5ec;padding:10px}</style>'
        '<h1>Dspalarefiner_tool_v0 analysis</h1>'
        '<p><a href="predictor.html">Open the equation predictor</a></p>'
        + equation_section +
        '<h2>Predictive performance</h2><table><tr><th>Dataset</th><th>Metric</th><th>Original</th>'
        f'<th>{expression_label}</th><th>{delta_label}</th></tr>'
        + "".join(rows) + '</table>'
        '<h2>Approximation fidelity</h2><table><tr><th>Dataset</th><th>Measure</th><th>Value</th></tr>'
        + "".join(fidelity_rows) + '</table>'
        '<img src="analysis.png" alt="Performance and feature importance">'
        + original_shap_image + monitoring_section +
        '<h2>Saved artifacts</h2><p><a href="equation.json">Equation JSON</a> · '
        '<a href="encoder.json">Encoder JSON</a> · <a href="summary.json">Results JSON</a></p>'
        f'<details><summary>Full results, audits and provenance</summary><pre>{text}</pre></details></html>', encoding="utf-8")


def write_predictor(output, artifact, encoder, labels):
    payload = json.dumps({"equation": artifact, "encoder": encoder, "labels": labels},
                         allow_nan=False).replace("<", "\\u003c")
    template = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Explicit equation predictor</title>
<style>
body{font:16px system-ui;color:#19344b;background:#f5f8fb;max-width:1100px;margin:32px auto;padding:16px}
math{font-size:1.25em;margin:18px 0}.equation-math{overflow-x:auto}
section{background:white;padding:24px;border-radius:12px;margin:18px 0}
#fields{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
label{display:flex;flex-direction:column;gap:6px}input,select,textarea,button{font:inherit;padding:10px;border:1px solid #bdccd5;border-radius:6px}
button{background:#116b91;color:white;cursor:pointer}textarea{width:95%;height:120px}
.bar{display:flex;gap:12px;align-items:center;margin:8px 0}.track{height:10px;border-radius:5px}
#result{display:flex;align-items:baseline;gap:12px 24px;flex-wrap:wrap}.prediction-label{font-size:36px;font-weight:700}.normalized-score{font-size:20px;font-weight:500}small{color:#567}pre{white-space:pre-wrap}
</style>
<h1>Predict with an explicit quadratic equation</h1>
<section><h2>Input features</h2><div id="fields"></div>
<p><button id="predict">Calculate prediction</button></p><div id="result"></div>
<p id="scoreRule"></p>
<p id="inputError" role="alert" style="color:#ad3939"></p>
<div id="contributions"></div><div id="sampleDrill"></div></section>
<section><h2>Batch prediction</h2>
<p><button id="downloadTemplate">Download CSV template</button></p>
<small>The template includes one illustrative row. Replace it with your own values.</small>
<p><input id="batchFile" type="file" accept=".csv,text/csv" hidden><button id="uploadBatch">Upload CSV</button> <span id="batchFilename">No file selected</span></p>
<textarea id="batch"></textarea><p><button id="calculateBatch">Download predictions</button></p><p id="error" role="alert"></p></section>
<script>
const DATA=__DATA__;
function templateCSV(){
 const specs=DATA.encoder.spec;
 const headers=specs.map(s=>s.name);
 const example=specs.map(s=>{
  if(s.kind!=="numeric")return s.levels[0];
  const j=DATA.encoder.names.indexOf(s.name),v=DATA.equation.mean[j];
  return Number.isFinite(v)?Number(v.toPrecision(7)):0;
 });
 const quote=v=>'"'+String(v??"").replaceAll('"','""')+'"';
 return [headers,example].map(row=>row.map(quote).join(",")).join("\r\n");
}
function validateBatchCSV(text){
 const rows=parseCSV(text);
 for(const s of DATA.encoder.spec)if(!(s.name in rows[0]))throw Error("Missing CSV column: "+s.name);
 rows.forEach((row,i)=>{try{validateCompleteRow(row)}catch(e){throw Error("Row "+(i+2)+": "+e.message)}});
 return rows;
}
function validateCompleteRow(row){
 for(const s of DATA.encoder.spec){
  const value=row[s.name];
  if(value===undefined||value===null||String(value).trim()==="")throw Error("Required: "+s.name);
 }
 const values=encode(row);
 if(values.some(v=>!Number.isFinite(v)))throw Error("Select a valid value for every categorical feature.");
}
function encode(row){
 const values=[];
 for(const s of DATA.encoder.spec){
  const raw=row[s.name], blank=raw===undefined||raw===null||String(raw).trim()==="";
  if(s.kind==="numeric"){
   const n=blank?NaN:Number(raw);if(!blank&&!Number.isFinite(n))throw Error("Invalid number: "+s.name);
   values.push(n);
  }else{
   let key=String(raw);
   if(!blank&&s.numeric_codes){
    const n=Number(raw);
    key=s.levels.find(l=>l===String(raw)||(Number.isFinite(n)&&
     (!Number.isInteger(n)||Number.isSafeInteger(n))&&Number(l)===n));
   }
   for(const l of s.levels)values.push(blank||!s.levels.includes(key)?NaN:Number(key===l));
  }
 }return values;
}
function evaluate(row){
 const a=DATA.equation,x=encode(row),z=x.map((v,j)=>(v-a.mean[j])/a.std[j]);
 const terms=a.classes.map(e=>{
  if(a.kind!=="native_taylor")return z.map((v,j)=>Number.isFinite(v)?e.A[j]*v*v+e.B[j]*v:e.D[j]);
  const p=z.length,u=[...z.map(v=>Number.isFinite(v)?v:0),...z.map(v=>Number.isFinite(v)?0:1)];
  const t=u.map((v,i)=>{
   let value=v*e.linear[i];
   for(let j=0;j<u.length;j++)value+=.5*v*e.H[i][j]*u[j];
   if(e.T)for(let j=0;j<u.length;j++)for(let k=0;k<u.length;k++)value+=v*e.T[i][j][k]*u[j]*u[k]/6;
   return value;
  });
  return t.slice(0,p).map((v,j)=>v+t[p+j]);
 });
 const scores=terms.map((t,c)=>a.classes[c].intercept+t.reduce((s,v)=>s+v,0));
 let score,prediction,explain;
 if(a.kind==="regression_fitted_quadratic"){score=scores[0];prediction=score;explain=terms[0]}
 else{const best=scores.indexOf(Math.max(...scores));prediction=DATA.labels[best];
 score=scores.length===2?scores[1]-scores[0]:scores[best];explain=scores.length===2?terms[1].map((v,j)=>v-terms[0][j]):terms[best]}
 const groups={};DATA.encoder.groups.forEach((g,j)=>{if(Number.isFinite(x[j]))groups[g]=(groups[g]||0)+explain[j]});
 const constant=a.kind!=="regression_fitted_quadratic"&&scores.length===2?
 a.classes[1].intercept-a.classes[0].intercept:a.classes[scores.indexOf(Math.max(...scores))].intercept;
 return {prediction,score,constant,normalized_score:a.kind==="regression_fitted_quadratic"?null:
 (scores.length===2?1/(1+Math.exp(-score)):null),contributions:groups};
}
function parseCSV(text){
 const rows=[];let row=[],cell="",quoted=false;
 for(let i=0;i<text.length;i++){let c=text[i];
 if(c==='"'){if(quoted&&text[i+1]==='"'){cell+='"';i++}else quoted=!quoted}
 else if(c===','&&!quoted){row.push(cell);cell=""}
 else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(cell);if(row.some(v=>v!==""))rows.push(row);row=[];cell=""}
 else cell+=c}
 if(quoted)throw Error("Unclosed CSV quote");if(cell||row.length){row.push(cell);rows.push(row)}
 if(rows.length<2)throw Error("Provide a header and at least one data row");
 const headers=rows.shift();if(new Set(headers).size!==headers.length)throw Error("Duplicate CSV headers");
 return rows.map((r,i)=>{if(r.length!==headers.length)throw Error("CSV width mismatch at row "+(i+2));return Object.fromEntries(headers.map((h,j)=>[h,r[j]]))})
}
for(const s of DATA.encoder.spec){
 const label=document.createElement("label");label.textContent=s.name;const input=document.createElement(s.kind==="numeric"?"input":"select");
 input.dataset.feature=s.name;input.required=true;if(s.kind==="numeric"){input.type="number";input.step="any"}else{
 const opt=document.createElement("option");opt.value="";opt.textContent="Select a value";input.append(opt);
 for(const l of s.levels){const o=document.createElement("option");o.value=l;o.textContent=l;input.append(o)}}
 label.append(input);document.querySelector("#fields").append(label);
}
document.querySelector("#predict").onclick=()=>{try{
 const row=Object.fromEntries([...document.querySelectorAll("[data-feature]")].map(e=>[e.dataset.feature,e.value]));
 validateCompleteRow(row);
 const r=evaluate(row),result=document.querySelector("#result");result.replaceChildren();
 const prediction=document.createElement("span");prediction.className="prediction-label";prediction.textContent="Prediction: "+r.prediction;result.append(prediction);
 if(r.normalized_score!==null){const score=document.createElement("span");score.className="normalized-score";score.textContent="Normalized score: "+r.normalized_score.toFixed(4);result.append(score)}
 document.querySelector("#scoreRule").textContent=r.normalized_score===null?"":
 "Normalized score > 0.5: class "+DATA.labels[1]+"; ≤ 0.5: class "+DATA.labels[0]+".";
 const target=document.querySelector("#contributions");target.replaceChildren();
 document.querySelector("#sampleDrill").replaceChildren();
 const a=DATA.equation,z=encode(row).map((v,j)=>(v-a.mean[j])/a.std[j]);
 const classIndex=DATA.labels.indexOf(r.prediction);
 const drillTerms=equationTerms(a,DATA.encoder.names,DATA.encoder.groups,classIndex<0?0:classIndex);
 const entries=Object.entries(r.contributions).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])),max=Math.max(...entries.map(e=>Math.abs(e[1])),1e-10);
 for(const [name,v] of entries){const d=document.createElement("div");d.className="bar";
 d.tabIndex=0;d.setAttribute("role","button");d.style.cursor="pointer";d.title="Inspect all terms containing "+name;
 d.onclick=()=>renderTermDrill(document.querySelector("#sampleDrill"),drillTerms,name,z);
 d.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();d.click()}};
 const t=document.createElement("span");t.textContent=name+" "+v.toFixed(3);t.style.width="220px";
 const b=document.createElement("span");b.className="track";b.style.width=(220*Math.abs(v)/max)+"px";b.style.background=v>=0?"#9a63bf":"#009e78";d.append(t,b);target.append(d)}
 document.querySelector("#inputError").textContent="";
 }catch(e){document.querySelector("#result").textContent="";document.querySelector("#contributions").replaceChildren();document.querySelector("#sampleDrill").replaceChildren();document.querySelector("#scoreRule").textContent="";document.querySelector("#inputError").textContent=e.message}};
document.querySelector("#calculateBatch").onclick=()=>{try{
 const rows=validateBatchCSV(document.querySelector("#batch").value);
 const quote=v=>'"'+String(v??"").replaceAll('"','""')+'"';
 const out=[["prediction","normalized_score"],...rows.map(row=>{const r=evaluate(row);return[r.prediction,r.normalized_score]})];
 const url=URL.createObjectURL(new Blob([out.map(r=>r.map(quote).join(",")).join("\n")],{type:"text/csv"}));
 const link=document.createElement("a");link.href=url;link.download="predictions.csv";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 document.querySelector("#error").textContent="Calculated "+rows.length+" rows.";
 }catch(e){document.querySelector("#error").textContent=e.message}};
document.querySelector("#downloadTemplate").onclick=()=>{
 const url=URL.createObjectURL(new Blob(["\ufeff"+templateCSV()],{type:"text/csv;charset=utf-8"}));
 const link=document.createElement("a");link.href=url;link.download="prediction_template.csv";link.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
};
document.querySelector("#uploadBatch").onclick=()=>document.querySelector("#batchFile").click();
document.querySelector("#batchFile").onchange=async event=>{
 const file=event.target.files[0];if(!file)return;
 try{
  const text=(await file.text()).replace(/^\uFEFF/,"");
  document.querySelector("#batch").value=text;
  document.querySelector("#batchFilename").textContent=file.name;
  const rows=validateBatchCSV(text);
  document.querySelector("#error").textContent="Ready: "+rows.length+" rows.";
 }catch(e){document.querySelector("#error").textContent=e.message}
};
</script></html>"""
    page = template.replace("__DATA__", payload).replace(
        '<section><h2>Input features</h2>',rendered_equation(artifact,encoder)+'<section><h2>Input features</h2>')
    if artifact["kind"] == "native_taylor":
        page = page.replace("Predict with an explicit quadratic equation",
                            f'Predict with an order-{artifact["approximation_order"]} Taylor equation')
    (output / "predictor.html").write_text(page, encoding="utf-8")
