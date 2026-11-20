(() => {
 let messages=[], transcript=[], busy=false, controller=null;
 const log=$("#chatMessages"), activity=$("#chatActivity");
 function render(node,text){
  const math=[];
  const protectedText=text.replace(/```[\s\S]*?```|`[^`\n]+`|\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+\$/g,s=>{
   if(s.startsWith("`"))return s;
   const id=math.length;math.push(s);return `MATHPLACEHOLDER${id}END`;
  });
  node.innerHTML=DOMPurify.sanitize(marked.parse(protectedText));
  const walker=document.createTreeWalker(node,NodeFilter.SHOW_TEXT), leaves=[];
  while(walker.nextNode())leaves.push(walker.currentNode);
  for(const leaf of leaves){
   if(leaf.parentElement.closest("code,pre"))continue;
   const re=/MATHPLACEHOLDER(\d+)END/g;let match,last=0,frag=document.createDocumentFragment(),found=false;
   while((match=re.exec(leaf.textContent))){
    found=true;frag.append(document.createTextNode(leaf.textContent.slice(last,match.index)));
    const raw=math[Number(match[1])],display=raw.startsWith("$$")||raw.startsWith("\\[");
    const span=document.createElement(display?"div":"span");span.className="chat-math";
    const trim=raw.startsWith("$")?(display?2:1):2;
    katex.render(raw.slice(trim,-trim),span,{displayMode:display,throwOnError:false,trust:false,strict:"ignore"});
    frag.append(span);last=re.lastIndex;
   }
   if(found){frag.append(document.createTextNode(leaf.textContent.slice(last)));leaf.replaceWith(frag)}
  }
 }
 function bubble(role,text){
  $("#chatWelcome")?.remove();const item=document.createElement("article");item.className="chat-message "+role;
  const label=document.createElement("strong");label.textContent=role==="user"?"You":"Assistant";
  const body=document.createElement("div");body.className="chat-body";render(body,text);item.append(label,body);log.append(item);log.scrollTop=log.scrollHeight;return {item,body};
 }
 function controls(on){busy=on;$("#ask").disabled=on;$("#newChat").disabled=on;$("#stopReply").hidden=!on}
 function context(){
  if(!$("#assistantContext").checked)return {};
  const c={};
  for(const [key,id] of Object.entries({epochs:"epochs",landmarks:"landmarks",tether:"tether",seed:"seed",batch_size:"batch_size",learning_rate:"learning_rate",test_fraction:"test_fraction"})){
   const el=$("#"+id);if(el&&el.value!==""&&Number.isFinite(Number(el.value)))c[key]=Number(el.value);
  }
  c.methods=typeof chosenMethods==="function"?chosenMethods():[];
  return {configuration:c,selected_predictors:featureControls.filter(x=>x.check?.checked).length};
 }
 async function send(){
  const q=$("#question").value.trim();if(!q||busy)return;
  const prior=messages.slice();bubble("user",q);transcript.push({role:"user",content:q});$("#question").value="";
  const reply=bubble("assistant","");let text="",lastRender=0,done=false;controller=new AbortController();controls(true);
  try{
   const payload={question:q,llm:$("#llm").checked,history:prior,context:context()};
   payload.selected_run=current||lastViewedRun||$("#previous").value||"";
   if($("#assistantContext").checked&&$("#previous").value)payload.run=$("#previous").value;
   const response=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json","X-Tabular-Token":TOKEN},body:JSON.stringify(payload),signal:controller.signal});
   if(!response.ok){const err=await response.json();throw Error(err.error||"Request failed")}
   const reader=response.body.getReader(),decoder=new TextDecoder();let buffer="";
   function event(e){
    if(e.type==="status")activity.textContent=e.stage+"…";
    if(e.type==="delta"){text+=e.text;if(Date.now()-lastRender>70){render(reply.body,text);lastRender=Date.now();log.scrollTop=log.scrollHeight}}
    if(e.type==="error")throw Error(e.message);
    if(e.type==="done"){
     done=true;const r=e.result;text=r.answer;render(reply.body,text);
     const meta=document.createElement("details"),head=document.createElement("summary");head.textContent="References and workflow";meta.append(head);
     const refs=document.createElement("p");refs.textContent=(r.sources||[]).map(x=>x.title).join(" · ")||"No reference";
     const stats=document.createElement("small");stats.textContent=`${r.mode} · ${r.workflow?.llm_calls||0} model calls · ${r.workflow?.retrieval_rounds||0} retrieval rounds`;meta.append(refs,stats);reply.item.append(meta);
     if(r.workflow?.result_lookup){const source=document.createElement("p");source.textContent="Read saved results · "+r.workflow.result_lookup.run_id+" · summary, coefficients and rankings";meta.append(source)}
     if(!r.error){messages=[...prior,{role:"user",content:q},{role:"assistant",content:text}].slice(-12)}
     transcript.push({role:"assistant",content:text});activity.textContent=r.error?"Reply failed. Check settings or try again.":"Ready";
    }
   }
   while(true){const chunk=await reader.read();buffer+=decoder.decode(chunk.value||new Uint8Array(),{stream:!chunk.done});let idx;while((idx=buffer.indexOf("\n"))>=0){const line=buffer.slice(0,idx);buffer=buffer.slice(idx+1);if(line.trim())event(JSON.parse(line))}if(chunk.done)break}
   if(buffer.trim())event(JSON.parse(buffer));if(!done)throw Error("Connection ended before the reply completed.");
  }catch(e){
   render(reply.body,text|| (e.name==="AbortError"?"Reply stopped.":e.message));
   activity.textContent=e.name==="AbortError"?"Stopped locally. The server may need a moment to finish its current step.":e.message;
  }finally{controls(false);controller=null;log.scrollTop=log.scrollHeight}
 }
 $("#ask").onclick=send;
 $("#question").onkeydown=e=>{if(e.key==="Enter"&&!e.shiftKey&&!e.isComposing){e.preventDefault();send()}};
 $("#stopReply").onclick=()=>controller?.abort();
 $("#newChat").onclick=()=>{messages=[];transcript=[];log.replaceChildren();activity.textContent="New conversation";$("#question").focus()};
 $("#downloadChat").onclick=()=>{
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([transcript.map(m=>`## ${m.role==="user"?"You":"Assistant"}\n\n${m.content}`).join("\n\n")],{type:"text/markdown;charset=utf-8"}));a.download="assistant-chat.md";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
 };
 for(const b of document.querySelectorAll("[data-prompt]"))b.onclick=()=>{$("#question").value=b.dataset.prompt;$("#question").focus()};
 $("#llm").onchange=()=>{$("#chatMode").textContent=$("#llm").checked?"LLM enabled":"Local guide"};
 let discoveryVersion=0;
 const invalidateModels=()=>{discoveryVersion++;$("#llmModelOptions").replaceChildren();$("#llmConfigStatus").textContent="Connection changed. Discover models for this endpoint."};
 for(const id of ["llmBase","llmProvider","llmProtocol","llmKey","llmHttp"])$("#"+id).addEventListener("input",invalidateModels);
 async function diagnostic(path,button){
  const discovering=path==="/api/llm/models",version=discoveryVersion;
  button.disabled=true;$("#llmConfigStatus").textContent=discovering?"Discovering models from the current connection…":"Checking saved settings…";
  const payload=discovering?{connection:{provider:$("#llmProvider").value,protocol:$("#llmProtocol").value,
   base_url:$("#llmBase").value.trim(),model:"",api_key:$("#llmKey").value,allow_http:$("#llmHttp").checked}}:{};
  if(discovering)$("#llmModelOptions").replaceChildren();
  try{const r=await request(path,payload);if(discovering&&version!==discoveryVersion)return;if(r.error)throw Error(r.error.message);
   if(r.models){const list=$("#llmModelOptions");list.replaceChildren();for(const id of r.models){const o=document.createElement("option");o.value=id;list.append(o)}$("#llmConfigStatus").textContent=r.models.length?"Available model IDs: "+r.models.join(", "):"No model IDs returned. Enter a Model ID manually."}
   else $("#llmConfigStatus").textContent=`Connected · ${r.model} · ${r.elapsed_seconds}s · ${r.reply}`;
  }catch(e){if(!discovering||version===discoveryVersion)$("#llmConfigStatus").textContent=e.message+(discovering?" You can also enter a Model ID manually.":"")}finally{button.disabled=false}
 }
 $("#discoverLlm").onclick=()=>diagnostic("/api/llm/models",$("#discoverLlm"));
 $("#checkLlm").onclick=()=>diagnostic("/api/llm/check",$("#checkLlm"));
})();
