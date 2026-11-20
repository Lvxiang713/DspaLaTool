"""Loopback-only workflow UI. No remote hosting or background auto-training."""
import argparse
import base64
import json
import secrets
import subprocess
import sys
import threading
import time
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .assistant.service import ask_assistant, safe_context, run_context
from .data import read_table

PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Dspalarefiner_tool_v0</title>
<style>
body{font:15px system-ui;background:#f4f7fb;color:#173248;max-width:1150px;margin:28px auto;padding:12px}
h1{font-size:32px;margin-bottom:5px}h2{font-size:20px}p{line-height:1.5}
section{background:white;border:1px solid #e0e8ef;border-radius:14px;padding:24px;margin:18px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px}
[hidden]{display:none!important}label{display:flex;flex-direction:column;gap:7px}input,select,textarea,button{font:inherit;padding:10px;border:1px solid #bccdd8;border-radius:6px}
button{background:#087ba4;color:white;cursor:pointer}button:disabled{opacity:.5}
textarea{width:95%;height:80px}pre{white-space:pre-wrap;background:#f4f7fa;padding:16px}
.steps{display:flex;gap:8px;flex-wrap:wrap}.steps span{padding:10px;background:#edf4f8;border-radius:5px}
#progress{height:12px;background:#e1e9ef;border-radius:7px;overflow:hidden}#fill{background:#009e78;height:100%;width:0}
a{color:#087ba4}canvas{width:100%;height:180px}small{color:#5b6f80}
.upload{padding:22px;background:#f2f8fb;border:1px dashed #8bb8ca;border-radius:10px}
.feature-wrap{overflow:auto;max-height:360px;border:1px solid #e0e8ef;border-radius:8px;margin-top:14px}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:12px;border-bottom:1px solid #e5edf2}
th{background:#f3f7fa;position:sticky;top:0}td input[type=checkbox]{accent-color:#087ba4;width:17px;height:17px}
tr.target-row{background:#eef8f4;color:#517467}.toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:14px}
.secondary{background:white;color:#087ba4;border-color:#b9d3df}summary{cursor:pointer;font-weight:600;padding:12px 0}
.badge{display:inline-block;background:#eaf5f9;padding:5px 10px;border-radius:20px;font-size:13px}
.file-picker{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:8px}
.file-name{color:#526b7b;overflow-wrap:anywhere}.file-picker input[type=file]{display:none}
#parameterTip{position:fixed;z-index:1000;max-width:330px;padding:10px 13px;background:#173248;color:white;border-radius:7px;font-size:13px;line-height:1.45;box-shadow:0 4px 18px #17324833;pointer-events:none}
label[ data-help],div[data-help]{position:relative}
.help-mark{position:absolute;right:3px;top:0;font-size:12px;color:#087ba4;cursor:help}
#links{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.result-card{display:flex;align-items:stretch;flex-direction:column;gap:12px;padding:14px;background:#eef6fa;border-radius:10px;margin:12px 0}
.result-card button{font-weight:600;padding:13px 20px}.result-card a{font-size:13px}
#resultPreview{width:100%;height:720px;border:1px solid #d5e1e9;border-radius:10px;background:white}
#jsonPreview{max-height:650px;overflow:auto}
.method-choices{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:10px}
.method-choices label{border:1px solid #d7e3eb;border-radius:8px;padding:10px;background:#fafcfd;font-size:13px}
.method-choices label:has(input:checked){border-color:#087ba4;background:#edf8fc}
.method-choices input{accent-color:#087ba4;margin-right:6px}
.output-field{margin-top:18px}.output-field input{width:100%;box-sizing:border-box}
</style>
<h1>Dspalarefiner_tool_v0</h1><p>From tabular data to a trained model and an interpretable equation.</p>
<div class="steps"><span>1 · Prepare</span><span>2 · Train</span><span>3 · Integrate parameters</span><span>4 · Analyze</span><span>5 · Export HTML</span></div>
<section><h2>1 · Upload your dataset</h2>
<div class="upload"><span>Training CSV</span><div class="file-picker">
<input id="file" type="file" accept=".csv" aria-label="Training CSV">
<button id="chooseFile" type="button">Choose CSV file</button>
<span id="fileName" class="file-name" aria-live="polite">No file selected</span></div>
<p id="datasetSummary">Choose a CSV to detect columns and feature types automatically.</p></div>
</section>
<section><h2>2 · Choose the target and modeling variables</h2><div class="grid">
<label title="The column containing the class labels to predict. It is excluded from the modeling variables.">Prediction target<select id="target" disabled><option value="">Upload a CSV first</option></select></label>
<label title="This version supports classification in the interface, including binary and multiclass targets.">Task<select id="task"><option value="classification">Classification</option></select></label>
</div>
<div class="toolbar"><button id="selectAll" class="secondary" type="button">Select all predictors</button>
<button id="selectNone" class="secondary" type="button">Clear selection</button><span id="selectionCount" class="badge">No dataset loaded</span></div>
<div class="feature-wrap"><table><thead><tr><th>Use</th><th>Variable</th><th>Feature type</th><th>Unique values</th></tr></thead>
<tbody id="featureRows"><tr><td colspan="5">Detected features will appear here.</td></tr></tbody></table></div>
<p><small>The target is automatically excluded from predictors. Deselect identifiers or variables unavailable at prediction time. Types shown here are suggestions; encoding is fitted on training rows only.</small></p>
<h3>Basic settings</h3><div class="grid">
<label title="Choose CPU or a CUDA-capable GPU. CUDA requires compatible hardware and a working PyTorch CUDA installation.">Device<select id="device"><option value="cpu">CPU</option><option value="cuda">CUDA GPU</option></select></label>
<label title="Enabled computes and exports SHAP feature contributions for both the original model and its quadratic expression. Disabled skips both. This adds computation time and storage.">SHAP analysis<select id="original_shap"><option value="yes">Enabled</option><option value="no">Disabled</option></select></label>
<div title="Select multiple expressions to compare them using one trained model. Taylor uses source-standardized zero as the reference; third order supports at most 40 encoded features."><span>Equation extraction (select one or more)</span>
<div class="method-choices">
<label><span><input type="checkbox" name="method" value="geometric" checked> Geometric additive</span></label>
<label><span><input type="checkbox" name="method" value="taylor1"> Taylor · first order (linear)</span></label>
<label><span><input type="checkbox" name="method" value="taylor2"> Taylor · second order (Hessian)</span></label>
<label><span><input type="checkbox" name="method" value="taylor3"> Taylor · third order (cubic)</span></label></div></div>
</div>
<label class="output-field" title="A new run folder is created under this directory. Leave blank to save in the application's default results folder.">Output directory<input id="output_directory" type="text" placeholder="Optional custom output folder"><small>Leave blank to use the default results folder.</small></label>
<details id="advanced"><summary>Advanced settings</summary>
<div class="grid">
<div title="Optional independent dataset used only for evaluation after training. It is never used to fit preprocessing, train the model or select parameters."><span>External evaluation CSV (optional)</span><div class="file-picker">
<input id="external" type="file" accept=".csv" aria-label="External evaluation CSV">
<button id="chooseExternal" type="button" class="secondary">Choose external CSV</button>
<span id="externalName" class="file-name" aria-live="polite">No file selected</span></div></div>
<label title="Number of complete passes over the training data. Adjust this to control training duration.">Epochs<input id="epochs" type="number" min="1" max="10000" value="150"></label>
<label title="Number of learned reference landmarks for each class. Default: 128.">Landmarks per class<input id="landmarks" type="number" min="1" max="1024" value="128"></label>
<label title="Weight of the tether regularization term relative to classification loss. Keep the default of 1 in this version; changes are not recommended.">Tether weight<input id="tether" type="number" min="0" max="100" step=".1" value="1"><small>Recommended: keep 1 in this version.</small></label>
<label title="Controls reproducible data splitting and model initialization. Change it to run another reproducible initialization.">Random seed<input id="seed" type="number" min="0" value="0"></label>
<label title="Fraction of uploaded rows reserved for evaluation and excluded from training. For example, 0.25 reserves 25%. It does not select the trained model.">Holdout fraction<input id="test_fraction" type="number" min=".05" max=".5" step=".05" value=".25"></label>
<label title="Number of rows processed together in each training batch. Gradients accumulate across batches before the epoch update. Smaller batches reduce temporary memory usage. Default: 8.">Batch size<input id="batch_size" type="number" min="1" max="100000" value="8"></label>
<label title="Initial optimizer step size controlling parameter updates. Keep the default of 0.01 in this version; changes are not recommended.">Learning rate<input id="learning_rate" type="number" min=".000001" max="1" step=".001" value=".01"><small>Recommended: keep 0.01 in this version.</small></label>
</div><p><small>Training parameters apply to the native DSpaLa classifier.</small></p></details>
<p><button id="start" disabled>Start modeling and analysis</button></p>
<p id="error" role="alert"></p></section>
<section><h2>Reanalyze a saved model</h2>
<div class="toolbar"><select id="analysisParent"><option value="">Choose a completed model</option></select>
<button id="reanalyze">Run analysis only</button></div>
</section>
<section><h2>Workflow and training monitor</h2><p id="state">Ready</p>
<p><select id="previous"><option value="">Choose a saved run</option></select>
<button id="refreshRuns">Refresh saved runs</button>
<button id="stop" disabled style="background:#ac3d37">Stop training</button></p>
<div id="progress"><div id="fill"></div></div>
<h3>Mean Total Loss</h3><canvas id="totalChart" width="1000" height="210" style="height:auto"></canvas>
<p id="savedPath" style="overflow-wrap:anywhere"></p>
<div id="links"></div>
<div id="previewPanel" hidden><h3 id="previewTitle">Result preview</h3>
<iframe id="resultPreview" title="Result preview"></iframe><pre id="jsonPreview" hidden></pre></div>
<pre id="details">No run started.</pre></section>
<section id="assistantPanel"><div class="chat-heading"><div><h2>Modeling assistant</h2><p>Ask, explore and follow up on your model and results.</p></div><span id="chatMode" class="badge">Local guide</span></div>
<details id="llmSettings"><summary>LLM settings</summary>
<form id="llmForm" autocomplete="off"><div class="grid">
<label>Service<select id="llmProvider" required><option value="">Select service</option><option value="deepseek">DeepSeek</option><option value="qwen">Qwen</option><option value="others">Others</option></select></label>
<label id="llmProtocolField" hidden>API protocol<select id="llmProtocol"><option value="">Select API protocol</option><option value="responses">Responses</option><option value="chat_completions">Chat Completions</option></select></label>
<label><span id="reasoningLabel">Thinking mode</span><select id="llmReasoning"><option value="off">Model default</option></select><small id="reasoningHelp">Checking model capability.</small></label>
<label id="llmBaseField" hidden>API base URL<input id="llmBase" type="url" placeholder="https://your-endpoint/v1" autocomplete="off"></label>
<label>Model ID<input id="llmModel" placeholder="Exact model ID" autocomplete="off" required></label>
<label>API key<input id="llmKey" type="password" placeholder="Enter key; leave blank to keep the saved key" autocomplete="new-password"></label>
</div><p id="llmServiceHelp"></p><p id="llmHttpField" hidden><label><span><input id="llmHttp" type="checkbox"> Allow HTTP for my trusted endpoint (requests and credentials are not TLS-encrypted)</span></label></p>
<div class="toolbar"><button type="submit" id="saveLlm">Save for this server session</button><button type="button" id="discoverLlm" class="secondary">Discover models</button><button type="button" id="checkLlm" class="secondary">Test connection</button><button type="button" id="clearLlm" class="secondary">Clear API key</button></div>
<datalist id="llmModelOptions"></datalist>
<p id="llmConfigStatus" aria-live="polite"></p>
<small>The key stays in server memory and is not written to files. Re-enter your connection settings after a server restart.</small>
</form></details>
<p><label><span><input id="assistantContext" type="checkbox"> Include current form settings. Selected run results are read automatically.</span></label></p>
<p><label><span><input id="llm" type="checkbox"> Use Qwen / DeepSeek LLM for this question</span></label></p>
<div class="chat-suggestions"><button type="button" class="secondary" data-prompt="How do I start modeling?">Getting started</button><button type="button" class="secondary" data-prompt="Explain the geometric quadratic expression and Taylor approximation.">Compare expressions</button><button type="button" class="secondary" data-prompt="Help me interpret the selected run's results.">Interpret results</button></div>
<div id="chatMessages" role="log" aria-label="Conversation"><div id="chatWelcome">Ready when you are. Ask about settings, equations or results.<br><small>Follow-up questions use this conversation. No training actions are performed.</small></div></div>
<div id="chatActivity" role="status" aria-live="polite">Ready</div>
<div class="chat-compose"><textarea id="question" rows="2" placeholder="Ask a question… Enter to send · Shift+Enter for a new line"></textarea><button id="ask">Send</button><button id="stopReply" class="secondary" hidden>Stop reply</button></div>
<div class="toolbar"><button id="newChat" class="secondary" type="button">New chat</button><button id="downloadChat" class="secondary" type="button">Download chat</button><small>Conversation stays in this page's memory. Max 3 model calls and 2 retrieval rounds per turn.</small></div>
</section>
<link rel="stylesheet" href="/assistant-assets/katex.min.css"><link rel="stylesheet" href="/assistant-assets/chat.css">
<script src="/assistant-assets/marked.min.js"></script><script src="/assistant-assets/purify.min.js"></script><script src="/assistant-assets/katex.min.js"></script>
<script>__MONITOR_JS__</script><script>
const TOKEN="__TOKEN__";let current=null,lastViewedRun="",history=[],featureControls=[],loadedFile=null;
const $=s=>document.querySelector(s);
const tip=document.createElement("div");tip.id="parameterTip";tip.hidden=true;document.body.append(tip);
for(const el of document.querySelectorAll("[title]")){
 const text=el.title;el.removeAttribute("title");el.setAttribute("data-help","");
 const mark=document.createElement("span");mark.className="help-mark";mark.textContent="ⓘ";mark.tabIndex=0;mark.setAttribute("aria-label",text);el.prepend(mark);
 const show=()=>{const r=el.getBoundingClientRect();tip.textContent=text;tip.hidden=false;
 tip.style.left=Math.max(8,Math.min(r.left,innerWidth-360))+"px";
 tip.style.top=Math.max(8,Math.min(r.bottom+7,innerHeight-tip.offsetHeight-10))+"px";};
 el.addEventListener("mouseenter",show);el.addEventListener("mouseleave",()=>tip.hidden=true);
 el.addEventListener("focusin",show);el.addEventListener("focusout",()=>tip.hidden=true);
}
$("#chooseFile").onclick=()=>$("#file").click();
function chosenMethods(){const m=[...document.querySelectorAll('input[name="method"]:checked')].map(e=>e.value);
 if(!m.length)throw Error("Select at least one equation extraction method.");return m}
$("#chooseExternal").onclick=()=>$("#external").click();
$("#external").onchange=()=>{$("#externalName").textContent=$("#external").files[0]?.name||"No file selected"};
async function request(path,data){const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json","X-Tabular-Token":TOKEN},body:JSON.stringify(data)});
 const obj=await r.json();if(!r.ok)throw Error(obj.error||r.statusText);return obj}
async function encodeFile(file){if(file.size>15000000)throw Error("Choose a CSV smaller than 15 MB.");
 const bytes=new Uint8Array(await file.arrayBuffer());let binary="";for(const b of bytes)binary+=String.fromCharCode(b);return btoa(binary)}
function updateSelection(){
 const target=$("#target").value;
 for(const item of featureControls){const isTarget=item.name===target;item.check.disabled=isTarget;
 item.type.disabled=isTarget;item.row.classList.toggle("target-row",isTarget);}
 const n=featureControls.filter(i=>!i.check.disabled&&i.check.checked).length;
 $("#selectionCount").textContent=n+" predictors selected";
 $("#start").disabled=!loadedFile||!target||n===0||!!current;
 $("#reanalyze").disabled=!!current;
}
$("#target").onchange=updateSelection;
$("#selectAll").onclick=()=>{featureControls.forEach(i=>{if(!i.check.disabled)i.check.checked=true});updateSelection()};
$("#selectNone").onclick=()=>{featureControls.forEach(i=>{if(!i.check.disabled)i.check.checked=false});updateSelection()};
$("#file").onchange=async()=>{loadedFile=null;featureControls=[];$("#start").disabled=true;$("#target").disabled=true;
 $("#fileName").textContent=$("#file").files[0]?.name||"No file selected";
 try{const file=$("#file").files[0];if(!file)return;
 $("#datasetSummary").textContent="Inspecting CSV…";
 const details=await request("/api/inspect",{csv:await encodeFile(file)});
 if($("#file").files[0]!==file)return;
 $("#target").replaceChildren();$("#featureRows").replaceChildren();
 const prompt=document.createElement("option");prompt.value="";prompt.textContent="Select the prediction target";$("#target").append(prompt);
 for(const col of details.features){
  const option=document.createElement("option");option.value=col.name;option.textContent=col.name;$("#target").append(option);
  const row=document.createElement("tr"),cells=Array.from({length:4},()=>document.createElement("td"));
  const check=document.createElement("input");check.type="checkbox";check.checked=true;check.setAttribute("aria-label","Use "+col.name);check.onchange=updateSelection;
  const type=document.createElement("select");type.setAttribute("aria-label","Type of "+col.name);
  type.title="Auto uses the detected type. Numeric retains a missing-value mask; categorical uses training-only category encoding.";
  check.title="Include this variable as a predictor. Exclude identifiers and information unavailable at prediction time.";
  for(const [value,text] of [["auto","Auto ("+col.inferred_type+")"],["numeric","Numeric"],["categorical","Categorical"]]){
   const o=document.createElement("option");o.value=value;o.textContent=text;type.append(o)}
  cells[0].append(check);cells[1].textContent=col.name;cells[2].append(type);
  cells[3].textContent=col.unique;
  row.append(...cells);$("#featureRows").append(row);featureControls.push({name:col.name,check,type,row});
 }
 loadedFile=file;$("#target").disabled=false;
 $("#datasetSummary").textContent=file.name+" · "+details.rows+" rows · "+details.columns.length+" columns";
 $("#error").textContent="";updateSelection();
 }catch(e){$("#datasetSummary").textContent="Could not inspect this CSV.";$("#error").textContent=e.message}};
$("#start").onclick=async()=>{try{const file=$("#file").files[0];if(!file)throw Error("Choose a CSV file.");
 if(!$("#target").value.trim())throw Error("Enter the target column.");
 if(file!==loadedFile)throw Error("Wait for CSV inspection before starting.");
 const selected=featureControls.filter(i=>i.name!==$("#target").value&&i.check.checked).map(i=>i.name);
 if(!selected.length)throw Error("Select at least one modeling variable.");
 const schema=Object.fromEntries(featureControls.filter(i=>i.name!==$("#target").value&&i.type.value!=="auto").map(i=>[i.name,i.type.value]));
 const external=$("#external").files[0];
 $("#start").disabled=true;
 const r=await request("/api/start",{csv:await encodeFile(file),external_csv:external?await encodeFile(external):null,feature_schema:schema,selected_features:selected,target:$("#target").value,
 task:$("#task").value,epochs:Number($("#epochs").value),landmarks:Number($("#landmarks").value),
 tether:Number($("#tether").value),seed:Number($("#seed").value),test_fraction:Number($("#test_fraction").value),batch_size:Number($("#batch_size").value),
 learning_rate:Number($("#learning_rate").value),dropout:0,device:$("#device").value,
 methods:chosenMethods(),output_directory:$("#output_directory").value.trim(),
 original_shap:$("#original_shap").value==="yes"});
 current=r.id;$("#start").disabled=true;$("#error").textContent="";poll();
 }catch(e){$("#error").textContent=e.message;updateSelection()}};
function draw(rows){TrainingView.loss($("#totalChart"),rows,"total_loss_mean","Mean Total Loss","#173248");}
$("#reanalyze").onclick=async()=>{try{const parent=$("#analysisParent").value;
 if(!parent)throw Error("Choose a completed model.");
 const r=await request("/api/reanalyze",{parent,methods:chosenMethods(),original_shap:$("#original_shap").value==="yes",output_directory:$("#output_directory").value.trim()});
 current=r.id;$("#start").disabled=true;$("#error").textContent="";poll();
 }catch(e){$("#error").textContent=e.message}};
$("#stop").onclick=async()=>{if(!current)return;const id=current;$("#stop").disabled=true;
 try{await request("/api/stop",{id});current=id;await poll()}catch(e){$("#error").textContent=e.message}};
async function poll(){if(!current)return;lastViewedRun=current;try{const r=await fetch("/api/status?id="+current),s=await r.json();
 $("#state").textContent=s.status.stage+" · "+(s.running?"running":s.status.status);
 $("#savedPath").textContent=s.output_directory?"Saved to: "+s.output_directory:"";
 const last=s.history.at(-1);const lossRow=last?{epoch:last.epoch,total_loss_mean:last.total_loss_mean}:null;
 $("#details").textContent=JSON.stringify({...s.status,latest_epoch:lossRow},null,2);draw(s.history);
 $("#stop").disabled=!s.running||!s.process_attached;
 $("#fill").style.width=(s.status.status==="complete"?100:Math.min(95,100*(s.history.at(-1)?.epoch||0)/s.epochs))+"%";
 $("#links").replaceChildren();
 if(s.status.status==="complete"){const links=$("#links");
 for(const [name,path] of [["Analysis report","report.html"],["Equation predictor","predictor.html"],["Results JSON","summary.json"]]){
 const url="/runs/"+current+"/"+path,card=document.createElement("div");card.className="result-card";
 const b=document.createElement("button");b.textContent="Preview "+name;b.onclick=()=>previewResult(name,url,path.endsWith(".json"));
 const a=document.createElement("a");a.textContent="Open in new tab ↗";a.href=url;a.target="_blank";card.append(b,a);links.append(card)}
 if($("#previewPanel").hidden)previewResult("Analysis report","/runs/"+current+"/report.html",false);}
 if(s.running||s.status.status==="unknown")setTimeout(poll,2000);else{current=null;updateSelection();listRuns();}
 }catch(e){$("#error").textContent=e.message;$("#start").disabled=false}}
async function previewResult(name,url,json){$("#previewPanel").hidden=false;$("#previewTitle").textContent=name;
 $("#resultPreview").hidden=json;$("#jsonPreview").hidden=!json;
 if(json){try{$("#jsonPreview").textContent=JSON.stringify(await(await fetch(url)).json(),null,2)}
 catch(e){$("#jsonPreview").textContent=e.message}}else if($("#resultPreview").getAttribute("src")!==url){$("#resultPreview").src=url}}
async function listRuns(){const r=await fetch("/api/runs"),runs=await r.json(),select=$("#previous"),selected=lastViewedRun||select.value;select.replaceChildren();
 const parents=$("#analysisParent"),selectedParent=parents.value;parents.replaceChildren();
 const prompt=document.createElement("option");prompt.value="";prompt.textContent="Choose a completed model";parents.append(prompt);
 const first=document.createElement("option");first.value="";first.textContent="Choose a saved run";select.append(first);
 for(const run of runs){const o=document.createElement("option");o.value=run.id;o.textContent=run.id+" · "+run.status;select.append(o);
 if(run.status==="complete"){const p=document.createElement("option");p.value=run.id;p.textContent=run.id;parents.append(p)}}
 select.value=selected;parents.value=selectedParent;
 if(!current){const active=runs.find(r=>r.status==="running");if(active){current=active.id;select.value=active.id;poll()}}}
$("#refreshRuns").onclick=listRuns;
$("#previous").onchange=()=>{if($("#previous").value){$("#previewPanel").hidden=true;current=$("#previous").value;poll()}};
listRuns().catch(e=>{$("#error").textContent=e.message});
const officialLlm={deepseek:"https://api.deepseek.com",qwen:"https://dashscope.aliyuncs.com/compatible-mode/v1"};
function serviceFields(preset=false){
 const service=$("#llmProvider").value,custom=service==="others";
 for(const id of ["llmProtocolField","llmBaseField","llmHttpField"])$("#"+id).hidden=!custom;
 $("#llmProtocol").required=custom;$("#llmBase").required=custom;
 if(preset){$("#llmBase").value=officialLlm[service]||"";$("#llmProtocol").value=officialLlm[service]?"chat_completions":"";$("#llmHttp").checked=false}
 $("#llmServiceHelp").textContent=service==="qwen"?"Qwen official API · China (Beijing). For other regions or third-party services, select Others.":service==="deepseek"?"DeepSeek official API. Enter your API key, then discover and select a model.":"";
}
function showLlmConfig(c){
 const official=officialLlm[c.provider]&&c.base_url?.replace(/\/$/,"")===officialLlm[c.provider];
 $("#llmProvider").value=c.provider?(official?c.provider:"others"):"";$("#llmProtocol").value=c.protocol||"";
 $("#llmBase").value=c.base_url||"";$("#llmModel").value=c.model||"";
 serviceFields();
 reasoningOptions(c.reasoning_effort||"off");
 $("#llmHttp").checked=!!c.allow_http;$("#llmKey").value="";
 $("#llmConfigStatus").textContent=c.key_configured?"API key configured. Saved for this server session.":"No API key configured.";
 $("#llm").checked=!!c.key_configured;$("#llmModel").setAttribute("list","llmModelOptions");
 $("#chatMode").textContent=c.key_configured?"LLM ready":"Local guide";
}
let capabilityRequest=0;
async function reasoningOptions(value){
 const seq=++capabilityRequest,select=$("#llmReasoning");select.disabled=true;$("#saveLlm").disabled=true;
 if(!$("#llmProvider").value||!$("#llmModel").value.trim()){
 select.replaceChildren(new Option("Select a model first","off"));$("#reasoningLabel").textContent="Thinking mode";
 $("#reasoningHelp").textContent="";$("#saveLlm").disabled=false;return}
 try{const c=await request("/api/llm/capabilities",{provider:$("#llmProvider").value,model:$("#llmModel").value});
 if(seq!==capabilityRequest)return;
 select.replaceChildren();
 for(const level of c.levels){
 const o=document.createElement("option");o.value=level;o.textContent=c.kind==="unknown"?"Model default":level==="on"?"Enabled":level==="off"?(c.kind==="effort"?"Off · fast response":"Disabled"):level[0].toUpperCase()+level.slice(1);select.append(o)}
 select.value=c.levels.includes(value)?value:c.levels[0];select.disabled=c.levels.length<2;
 $("#reasoningLabel").textContent=c.label;$("#reasoningHelp").textContent=c.note;
 }catch(e){$("#reasoningHelp").textContent="Could not check model capability. Save is unavailable."}
 finally{if(seq===capabilityRequest)$("#saveLlm").disabled=$("#reasoningHelp").textContent.startsWith("Could not")}
}
$("#llmProvider").onchange=()=>{serviceFields(true);$("#llmModel").value="";$("#llmKey").value="";reasoningOptions("off")};
$("#llmModel").oninput=()=>reasoningOptions($("#llmReasoning").value);
$("#llmProtocol").onchange=()=>reasoningOptions($("#llmReasoning").value);
request("/api/llm/config/status",{}).then(showLlmConfig).catch(()=>{$("#llmConfigStatus").textContent="Could not load LLM settings."});
$("#llmForm").onsubmit=async e=>{e.preventDefault();$("#saveLlm").disabled=true;
 try{showLlmConfig(await request("/api/llm/config",{
 provider:$("#llmProvider").value,protocol:$("#llmProtocol").value,base_url:$("#llmBase").value.trim(),
 model:$("#llmModel").value.trim(),reasoning_effort:$("#llmReasoning").value,api_key:$("#llmKey").value,allow_http:$("#llmHttp").checked}));}
 catch(e){$("#llmKey").value="";$("#llmConfigStatus").textContent=e.message}
 finally{$("#saveLlm").disabled=false}};
$("#clearLlm").onclick=async()=>{try{showLlmConfig(await request("/api/llm/config/clear",{}))}
 catch(e){$("#llmConfigStatus").textContent=e.message}};
</script><script>__ASSISTANT_JS__</script></html>"""


class Application:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.jobs = {}
        self.lock = threading.Lock()
        self.chat_slot = threading.BoundedSemaphore(1)
        self.llm_config = {"provider":"","model":"","base_url":"","api_key":"",
                           "protocol":"","reasoning_effort":"off","allow_http":False}
        # Reopen persisted artifacts without treating a saved PID as a process.
        for directory in self.root.iterdir():
            record = directory / "job.json"
            if directory.is_symlink() or not record.is_file():
                continue
            try:
                meta = json.loads(record.read_text(encoding="utf-8"))
                if meta["id"] != directory.name:
                    continue
                self.jobs[directory.name] = {"process": None, "output": Path(meta.get("output",str(directory/"analysis"))).resolve(),
                                             "epochs": int(meta["epochs"])}
            except (ValueError,KeyError,TypeError,OSError):
                continue

    def output_path(self, obj, job):
        value=obj.get("output_directory","")
        if not isinstance(value,str) or len(value)>2000 or "\0" in value:
            raise ValueError("Enter a valid output directory.")
        base=Path(value.strip()).expanduser().resolve() if value.strip() else self.root
        if base.exists() and not base.is_dir():
            raise ValueError("Output directory points to a file.")
        return base/job/"analysis"

    def start(self, obj):
        with self.lock:
            for job_id in self.jobs:
                previous = self.status(job_id)
                if previous["running"] or previous["status"]["status"] == "unknown":
                    raise ValueError("An active or detached unfinished run exists. Wait for completion or inspect its process log before starting another run.")
            target = str(obj["target"]).strip()
            if not target:
                raise ValueError("A target column is required.")
            if obj["task"] not in ("classification", "regression"):
                raise ValueError("Unknown task.")
            approximation = obj.get("approximation","geometric")
            order = obj.get("approximation_order",2)
            from .analysis_batch import CHOICES
            methods = obj.get("methods")
            if methods is not None:
                if (not isinstance(methods,list) or not methods
                        or any(not isinstance(m,str) or m not in CHOICES for m in methods)
                        or len(set(methods))!=len(methods)):
                    raise ValueError("Select one or more unique extraction methods.")
                approximation,order = CHOICES[methods[0]]
            if approximation not in ("geometric","taylor") or isinstance(order,bool) or order not in (1,2,3):
                raise ValueError("Choose geometric or Taylor extraction, with Taylor order 1, 2 or 3.")
            for name, lower, upper in [("epochs",1,10000),("landmarks",1,1024),("seed",0,2**31-1)]:
                value = obj[name]
                if not isinstance(value, int) or isinstance(value,bool) or not lower <= value <= upper:
                    raise ValueError(f"Invalid {name}.")
            if not isinstance(obj["tether"],(float,int)) or not 0 <= obj["tether"] <= 100:
                raise ValueError("Invalid tether weight.")
            raw = base64.b64decode(obj["csv"], validate=True)
            if not raw or len(raw) > 15_000_000:
                raise ValueError("Choose a non-empty CSV smaller than 15 MB.")
            inspected = read_table(io.BytesIO(raw))
            if target not in inspected:
                raise ValueError("Target column is absent from the training CSV.")
            schema = obj.get("feature_schema") or {}
            if not isinstance(schema,dict) or any(v not in ("numeric","categorical","ignore") for v in schema.values()):
                raise ValueError("Feature schema must map names to numeric, categorical or ignore.")
            if target in schema or set(schema)-set(inspected.columns):
                raise ValueError("Feature schema may include predictor columns only, without the target.")
            schema = dict(schema)
            selected = obj.get("selected_features")
            if selected is not None:
                if (not isinstance(selected,list) or not selected
                        or any(not isinstance(name,str) for name in selected)
                        or len(set(selected)) != len(selected)
                        or target in selected or set(selected)-set(inspected.columns)):
                    raise ValueError("Select one or more unique predictor columns, excluding the target.")
                if any(schema.get(name) == "ignore" for name in selected):
                    raise ValueError("A selected feature cannot also be marked ignore.")
                for name in inspected.columns:
                    if name != target and name not in selected:
                        schema[name] = "ignore"
            advanced = {}
            for name, default, lower, upper in [
                ("test_fraction",.25,.05,.5), ("learning_rate",.01,.000001,1),
                ("dropout",0,0,.9), ("batch_size",8,1,100000)]:
                value = obj.get(name,default)
                if (isinstance(value,bool) or not isinstance(value,(int,float))
                        or not lower <= value <= upper
                        or name=="batch_size" and not isinstance(value,int)):
                    raise ValueError(f"Invalid {name}.")
                advanced[name] = value
            device = obj.get("device","cpu")
            if device not in ("cpu","cuda"):
                raise ValueError("Device must be cpu or cuda.")
            external_raw = None
            if obj.get("external_csv"):
                external_raw = base64.b64decode(obj["external_csv"],validate=True)
                if not external_raw or len(external_raw)>15_000_000:
                    raise ValueError("External CSV must be non-empty and smaller than 15 MB.")
                read_table(io.BytesIO(external_raw))
            job = time.strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
            directory = self.root / job
            out = self.output_path(obj,job)
            directory.mkdir()
            (directory / "job.json").write_text(json.dumps({"id":job,"epochs":obj["epochs"],"target":target,"output":str(out)}),encoding="utf-8")
            source = directory / "input.csv"
            source.write_bytes(raw)
            if external_raw is not None:
                (directory/"external.csv").write_bytes(external_raw)
            if schema:
                (directory/"schema.json").write_text(json.dumps(schema),encoding="utf-8")
            cmd = [sys.executable,"-m","tabular_agent",str(source),"--target",target,"--output",str(out),
                   "--task",obj["task"],"--epochs",str(obj["epochs"]),"--landmarks",str(obj["landmarks"]),
                   "--tether",str(obj["tether"]),"--seed",str(obj["seed"])]
            if external_raw is not None:
                cmd.extend(["--external",str(directory/"external.csv")])
            if schema:
                cmd.extend(["--schema",str(directory/"schema.json")])
            for name, value in advanced.items():
                cmd.extend(["--"+name.replace("_","-"),str(value)])
            cmd.extend(["--device",device])
            cmd.extend(["--approximation",approximation,"--approximation-order",str(order)])
            if obj.get("original_shap") is True:
                cmd.append("--original-shap")
            else:
                cmd.append("--no-shap")
            if methods and len(methods)>1:
                cfg = dict(advanced,task=obj["task"],epochs=obj["epochs"],landmarks=obj["landmarks"],
                           tether=obj["tether"],seed=obj["seed"],device=device,feature_schema=schema,
                           original_shap=obj.get("original_shap") is True,no_shap=obj.get("original_shap") is not True)
                request = directory/"analysis_request.json"
                request.write_text(json.dumps(dict(csv=str(source),target=target,output=str(out),
                    config=cfg,methods=methods,external=str(directory/"external.csv") if external_raw else None)),encoding="utf-8")
                cmd = [sys.executable,"-m","tabular_agent.analysis_batch",str(request)]
            log = (directory / "process.log").open("w",encoding="utf-8")
            try:
                proc = subprocess.Popen(cmd,cwd=Path(__file__).resolve().parents[1],stdout=log,
                                        stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            finally:
                log.close()
            self.jobs[job] = {"process":proc,"output":out,"epochs":obj["epochs"]}
            return job

    def reanalyze(self, obj):
        from .analysis_batch import CHOICES
        with self.lock:
            if any(self.status(k)["running"] or self.status(k)["status"]["status"]=="unknown" for k in self.jobs):
                raise ValueError("Wait for the active run to finish or stop it first.")
            parent_id = str(obj.get("parent",""))
            if parent_id not in self.jobs or self.status(parent_id)["status"]["status"]!="complete":
                raise ValueError("Choose a completed saved model.")
            methods = obj.get("methods")
            if (not isinstance(methods,list) or not methods or any(not isinstance(m,str) or m not in CHOICES for m in methods)
                    or len(set(methods))!=len(methods)):
                raise ValueError("Select one or more unique extraction methods.")
            parent = self.jobs[parent_id]["output"].resolve()
            summary = json.loads((parent/"summary.json").read_text(encoding="utf-8"))
            if "methods" in summary:
                parent = parent/summary["methods"][0]
                summary = json.loads((parent/"summary.json").read_text(encoding="utf-8"))
            cfg = summary["configuration"]
            # Resolve the original source even when the chosen model is itself a reanalysis.
            origin = parent
            while cfg.get("reuse_analysis"):
                candidate = Path(cfg["reuse_analysis"]).resolve()
                if not any(candidate.is_relative_to(v["output"].resolve()) for v in self.jobs.values()):
                    raise ValueError("Saved source is outside this UI's runs directory.")
                origin = candidate
                cfg = json.loads((origin/"summary.json").read_text(encoding="utf-8"))["configuration"]
            original_job = next((k for k,v in self.jobs.items()
                                 if origin.is_relative_to(v["output"].resolve())),None)
            if original_job is None:
                raise ValueError("Original source CSV is unavailable in the saved runs.")
            directory_source = self.root/original_job
            source = directory_source/"input.csv"
            original_summary = json.loads((origin/"summary.json").read_text(encoding="utf-8"))
            source_frame = read_table(source)
            # The original target is stored in its launch request or command metadata.
            meta = json.loads((directory_source/"job.json").read_text(encoding="utf-8"))
            target = meta.get("target")
            if not target:
                enc = json.loads((origin/"encoder.json").read_text(encoding="utf-8"))
                candidates = [c for c in source_frame.columns if c not in enc["columns"]]
                if len(candidates)!=1:
                    raise ValueError("This older run does not identify its target unambiguously.")
                target = candidates[0]
            job = time.strftime("%Y%m%d_%H%M%S")+"_"+secrets.token_hex(3)
            directory = self.root/job
            out = self.output_path(obj,job)
            directory.mkdir()
            (directory/"job.json").write_text(json.dumps({"id":job,"epochs":cfg["epochs"],"target":target,
                                                         "mode":"analysis_only","parent":parent_id,"output":str(out)}),encoding="utf-8")
            cfg = dict(cfg,original_shap=obj.get("original_shap") is True,
                       no_shap=obj.get("original_shap") is not True)
            ext = directory_source/"external.csv"
            req = directory/"analysis_request.json"
            req.write_text(json.dumps(dict(csv=str(source),target=target,output=str(out),
                config=cfg,methods=methods,external=str(ext) if ext.exists() else None,parent=str(parent))),encoding="utf-8")
            with (directory/"process.log").open("w",encoding="utf-8") as log:
                proc = subprocess.Popen([sys.executable,"-m","tabular_agent.analysis_batch",str(req)],
                    cwd=Path(__file__).resolve().parents[1],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            self.jobs[job]={"process":proc,"output":out,"epochs":cfg["epochs"]}
            return job

    def stop(self, job):
        """Stop only the child process owned by this server; retain all artifacts."""
        with self.lock:
            if job not in self.jobs:
                raise ValueError("Unknown run.")
            item = self.jobs[job]
            proc = item["process"]
            if proc is None:
                raise ValueError("This server does not own the process for this saved run.")
            if proc.poll() is not None:
                return self.status(job)
            out = item["output"]
            out.mkdir(parents=True, exist_ok=True)
            (out / "STOP_REQUEST.json").write_text(json.dumps({
                "requested_at":time.time(), "pid":proc.pid,
                "previous_status":self.status(job)["status"]}), encoding="utf-8")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            temp = out / "status.stop.tmp.json"
            temp.write_text(json.dumps({"stage":"stopped","status":"stopped",
                "message":"Stopped by user. Existing logs and artifacts are preserved; this run may be incomplete."}),encoding="utf-8")
            temp.replace(out / "status.json")
            return self.status(job)

    def status(self, job):
        if job not in self.jobs:
            raise ValueError("Unknown run.")
        item = self.jobs[job]
        path = item["output"] / "status.json"
        try:
            status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stage":"starting","status":"running"}
        except json.JSONDecodeError:
            status = {"stage":"updating","status":"running"}
        attached = item["process"] is not None
        running = attached and item["process"].poll() is None
        if not attached and status["status"] == "running":
            status = {"stage":"detached","status":"unknown","last_recorded_stage":status["stage"],
                      "message":"The UI restarted and cannot verify process ownership. Artifacts are preserved; no automatic restart. Status will update if the original process finishes."}
        elif not running and status["status"] == "running":
            status = {"stage":"process_exit","status":"error","message":"Process exited before final status. Inspect process.log."}
        hist = item["output"] / "monitor.jsonl"
        rows = []
        if hist.exists():
            for line in hist.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        from .monitoring import display_history
        training_n=None
        summary_path=item["output"]/"summary.json"
        if summary_path.exists():
            try:
                summary=json.loads(summary_path.read_text(encoding="utf-8"))
                if "methods" in summary:
                    summary=json.loads((item["output"]/summary["methods"][0]/"summary.json").read_text(encoding="utf-8"))
                training_n=summary.get("training_n")
            except (OSError,ValueError,KeyError,IndexError):
                pass
        rows=display_history(rows,training_n)
        return {"status":status,"running":running,"history":rows,"epochs":item["epochs"],
                "process_attached":attached,"output_directory":str(item["output"])}


def handler(app):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, body, content="application/json"):
            data = json.dumps(body).encode() if content=="application/json" else body
            self.send_response(code)
            self.send_header("Content-Type",content)
            self.send_header("Content-Length",str(len(data)))
            self.send_header("X-Content-Type-Options","nosniff")
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self):
            return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}",f"localhost:{self.server.server_port}")

        def do_GET(self):
            if not self.valid_host():
                return self.respond(403,{"error":"Loopback host required."})
            parsed = urlparse(self.path)
            if parsed.path == "/":
                import html
                page=PAGE.replace("__TOKEN__",app.token).replace("__MONITOR_JS__",Path(__file__).with_name("loss_view.js").read_text(encoding="utf-8")).replace("__ASSISTANT_JS__",Path(__file__).with_name("assistant").joinpath("web","chat.js").read_text(encoding="utf-8"))
                return self.respond(200,page.encode(),"text/html; charset=utf-8")
            if parsed.path.startswith("/assistant-assets/"):
                relative=parsed.path[len("/assistant-assets/"):]
                root=Path(__file__).with_name("assistant")/"web"
                path=root.joinpath(relative).resolve()
                if (not path.is_relative_to(root.resolve()) or not path.is_file()
                        or path.suffix not in (".js",".css",".woff2",".woff",".ttf")):
                    return self.respond(404,{"error":"Asset not found."})
                content={".js":"application/javascript; charset=utf-8",".css":"text/css; charset=utf-8",
                         ".woff2":"font/woff2",".woff":"font/woff",".ttf":"font/ttf"}[path.suffix]
                return self.respond(200,path.read_bytes(),content)
            if parsed.path == "/api/runs":
                return self.respond(200,[{"id":job,"status":app.status(job)["status"]["status"]}
                                         for job in reversed(list(app.jobs))])
            if parsed.path == "/api/status":
                from urllib.parse import parse_qs
                try:
                    return self.respond(200,app.status(parse_qs(parsed.query)["id"][0]))
                except (KeyError,ValueError):
                    return self.respond(404,{"error":"Unknown run."})
            if parsed.path.startswith("/runs/"):
                parts = unquote(parsed.path).split("/")
                allowed = ("report.html","predictor.html","summary.json","encoder.json","equation.json","analysis.png","original_model_SHAP.png","coefficients.csv","interaction_coefficients.csv","cubic_coefficients.csv","comparison.csv")
                from .analysis_batch import CHOICES
                if (len(parts) not in (4,5) or parts[2] not in app.jobs or parts[-1] not in allowed
                        or len(parts)==5 and parts[3] not in CHOICES):
                    return self.respond(404,{"error":"Artifact not found."})
                path=app.jobs[parts[2]]["output"].joinpath(*parts[3:])
                if not path.is_file():
                    return self.respond(404,{"error":"Artifact not ready."})
                content="image/png" if path.suffix==".png" else "text/html; charset=utf-8" if path.suffix==".html" else "text/csv; charset=utf-8" if path.suffix==".csv" else "application/json; charset=utf-8"
                return self.respond(200,path.read_bytes(),content)
            self.respond(404,{"error":"Not found."})

        def do_POST(self):
            if not self.valid_host() or self.headers.get("X-Tabular-Token") != app.token:
                return self.respond(403,{"error":"Invalid local session."})
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0 < length <= 42_000_000:
                    raise ValueError("Request too large or empty.")
                obj=json.loads(self.rfile.read(length))
                if self.path=="/api/inspect":
                    raw=base64.b64decode(obj["csv"],validate=True)
                    if len(raw)>15_000_000:
                        raise ValueError("Choose a CSV smaller than 15 MB.")
                    frame=read_table(io.BytesIO(raw))
                    from pandas.api.types import is_numeric_dtype
                    features = [{"name":name,
                                 "inferred_type":"numeric" if is_numeric_dtype(frame[name]) else "categorical",
                                 "missing":int(frame[name].isna().sum()),
                                 "unique":int(frame[name].nunique())}
                                for name in frame.columns]
                    return self.respond(200,{"columns":list(frame.columns),"rows":len(frame),
                                             "features":features})
                if self.path=="/api/start":
                    return self.respond(200,{"id":app.start(obj)})
                if self.path=="/api/reanalyze":
                    return self.respond(200,{"id":app.reanalyze(obj)})
                if self.path=="/api/stop":
                    return self.respond(200,app.stop(str(obj.get("id",""))))
                if self.path=="/api/llm/capabilities":
                    from .assistant.capabilities import reasoning_capability
                    return self.respond(200,reasoning_capability(str(obj.get("provider","")),str(obj.get("model",""))))
                if self.path in ("/api/llm/models","/api/llm/check"):
                    from .assistant.provider import ChatProvider, ProviderError
                    with app.lock:
                        config=dict(app.llm_config)
                    if self.path=="/api/llm/models":
                        # Discovery uses the unsaved form. Never send a saved key
                        # to a different endpoint without an explicitly supplied key.
                        supplied=obj.get("connection")
                        if not isinstance(supplied,dict):
                            raise ValueError("Enter the current connection settings.")
                        candidate={k:supplied.get(k,"") for k in ("provider","protocol","base_url","model")}
                        candidate["allow_http"]=supplied.get("allow_http") is True
                        key=supplied.get("api_key","")
                        if not isinstance(key,str):
                            raise ValueError("Enter a valid API key.")
                        same_base=str(candidate["base_url"]).strip().rstrip("/")==str(config.get("base_url","")).strip().rstrip("/")
                        candidate["api_key"]=key or (config.get("api_key","") if same_base else "")
                        if not candidate["api_key"]:
                            raise ValueError("Enter an API key for this endpoint before discovering models.")
                        candidate["reasoning_effort"]="off"
                        config=candidate
                    provider=ChatProvider(config,discovery=self.path=="/api/llm/models")
                    try:
                        if self.path=="/api/llm/models":
                            return self.respond(200,{"models":provider.models()})
                        started=time.monotonic()
                        text=provider.call("Reply exactly CONNECTED.",{"question":"Connectivity check."},128)
                        return self.respond(200,{"ok":True,"model":provider.model,
                            "elapsed_seconds":round(time.monotonic()-started,2),"reply":text[:300]})
                    except ProviderError as exc:
                        return self.respond(200,{"ok":False,"error":exc.public()})
                    finally:
                        provider.close()
                if self.path in ("/api/llm/config/status", "/api/llm/config", "/api/llm/config/clear"):
                    from .assistant.provider import ChatProvider
                    with app.lock:
                        if self.path=="/api/llm/config":
                            if not isinstance(obj.get("api_key",""),str):
                                raise ValueError("Enter a valid API key.")
                            candidate={k:obj.get(k) for k in ("provider","model","base_url","protocol")}
                            if any(not isinstance(v,str) for v in candidate.values()):
                                raise ValueError("Complete LLM settings.")
                            candidate["allow_http"]=obj.get("allow_http") is True
                            candidate["reasoning_effort"]=obj.get("reasoning_effort",app.llm_config.get("reasoning_effort","off"))
                            candidate["api_key"]=obj.get("api_key") or app.llm_config.get("api_key","")
                            checked=ChatProvider(candidate)
                            checked.close()
                            app.llm_config=candidate
                        elif self.path=="/api/llm/config/clear":
                            app.llm_config={**app.llm_config,"api_key":""}
                        public={k:v for k,v in app.llm_config.items() if k!="api_key"}
                        public["key_configured"]=bool(app.llm_config.get("api_key"))
                    return self.respond(200,public)
                if self.path in ("/api/ask","/api/chat"):
                    q=str(obj.get("question","")).strip()
                    if not q or len(q)>8000:
                        raise ValueError("Enter a question of 1–8000 characters.")
                    context=safe_context(obj.get("context",{}))
                    from .assistant.service import asks_training_progress, live_training_context
                    from .assistant.service import resolve_result_run
                    from .assistant.service import safe_history
                    messages=safe_history(obj.get("history",[]))
                    result_run=resolve_result_run(app,q,str(obj.get("selected_run") or obj.get("run") or "") or None,
                                                  messages)
                    from .assistant.service import asks_training_loss
                    if asks_training_progress(q) or asks_training_loss(q):
                        context["live_training"]=live_training_context(app,result_run)
                    if result_run:
                        context["saved_run"]=run_context(app,result_run,q)
                    with app.lock:
                        provider_config=dict(app.llm_config)
                    if not app.chat_slot.acquire(blocking=False):
                        return self.respond(409,{"error":"An assistant reply is already in progress. Please wait."})
                    if self.path=="/api/ask":
                        try:
                            return self.respond(200,ask_assistant(q,obj.get("llm") is True,
                                context,provider_config,history=messages))
                        finally:
                            app.chat_slot.release()
                    self.send_response(200)
                    self.send_header("Content-Type","application/x-ndjson; charset=utf-8")
                    self.send_header("Cache-Control","no-store")
                    self.send_header("Connection","close")
                    self.end_headers()
                    self.close_connection=True
                    stream_lock=threading.Lock()
                    stream_closed=False
                    def emit(event):
                        nonlocal stream_closed
                        with stream_lock:
                            if stream_closed:
                                return
                            try:
                                self.wfile.write((json.dumps(event,ensure_ascii=False)+"\n").encode("utf-8"))
                                self.wfile.flush()
                            except OSError:
                                stream_closed=True
                                raise
                            if event.get("type") in ("done","error"):
                                stream_closed=True
                    try:
                        emit({"type":"status","stage":"Starting reply"})
                        result=ask_assistant(q,obj.get("llm") is True,context,provider_config,
                                             history=messages,emit=emit)
                        emit({"type":"done","result":result})
                    except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):
                        pass
                    except Exception:
                        try:
                            emit({"type":"error","message":"Could not complete the reply. Check LLM settings and try again."})
                        except OSError:
                            pass
                    finally:
                        with stream_lock:
                            stream_closed=True
                        app.chat_slot.release()
                    return
                self.respond(404,{"error":"Not found."})
            except Exception as exc:
                self.respond(400,{"error":str(exc)})

        def log_message(self,*args):
            pass
    return Handler


def main():
    p=argparse.ArgumentParser(description="Start the local English Dspalarefiner_tool_v0 UI")
    p.add_argument("--port",type=int,default=8765)
    p.add_argument("--runs",default="runs/ui")
    p.add_argument("--open-browser", action="store_true")
    args=p.parse_args()
    app=Application(args.runs)
    server=ThreadingHTTPServer(("127.0.0.1",args.port),handler(app))
    print(f"Open http://127.0.0.1:{server.server_port} · Press Ctrl+C to stop the UI.",flush=True)
    if args.open_browser:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=="__main__":
    main()
