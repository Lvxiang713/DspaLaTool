function equationTerms(a,names,groups,classIndex){
 const binary=a.classes.length===2&&a.kind!=="regression_fitted_quadratic";
 const e=a.classes[binary?1:(classIndex===undefined?a.classes.length-1:classIndex)];
 const base=binary?a.classes[0]:null,p=names.length,out=new Map();
 const add=(ix,c)=>{
  if(!c)return;
  ix=ix.slice().sort((x,y)=>x-y);
  const key=ix.join(","),old=out.get(key);
  if(old){old.coefficient+=c;return}
  const powers=new Map();ix.forEach(i=>powers.set(i,(powers.get(i)||0)+1));
  const label=[...powers].map(([i,k])=>"z["+names[i]+"]"+(k===2?"²":k===3?"³":"")).join(" × ");
  out.set(key,{indices:ix,names:ix.map(i=>names[i]),groups:ix.map(i=>groups[i]),label,coefficient:c});
 };
 if(a.kind==="native_taylor"){
  for(let i=0;i<p;i++){
   add([i],e.linear[i]-(base?base.linear[i]:0));
   for(let j=0;j<p;j++){
    add([i,j],(e.H[i][j]-(base?base.H[i][j]:0))/2);
    if(e.T)for(let k=0;k<p;k++)add([i,j,k],(e.T[i][j][k]-(base?base.T[i][j][k]:0))/6);
   }
  }
 }else for(let i=0;i<p;i++){
  add([i],e.B[i]-(base?base.B[i]:0));
  add([i,i],e.A[i]-(base?base.A[i]:0));
 }
 return [...out.values()].filter(t=>t.coefficient!==0);
}
function termMath(term){
 const ns="http://www.w3.org/1998/Math/MathML";
 const node=(tag,text)=>{const el=document.createElementNS(ns,tag);if(text!==undefined)el.textContent=text;return el};
 const math=node("math"),row=node("mrow"),powers=new Map();
 term.indices.forEach((index,k)=>{const current=powers.get(index);if(current)current.power++;else powers.set(index,{name:term.names[k],power:1})});
 let first=true;
 for(const {name,power} of powers.values()){
  if(!first)row.append(node("mo","·"));first=false;
  const variable=node("msub");variable.append(node("mi","z"),node("mtext",name));
  if(power>1){const raised=node("msup");raised.append(variable,node("mn",String(power)));row.append(raised)}
  else row.append(variable);
 }
 math.append(row);math.setAttribute("aria-label",term.label);
 math.style.cssText="font-size:1.1em;margin:0;max-width:100%";
 return math;
}
function renderTermDrill(target,terms,feature,z){
 target.replaceChildren();
 const title=document.createElement("h4");title.textContent=feature+" · "+(z?"sample contribution":"coefficient")+" breakdown";target.append(title);
 const rows=terms.filter(t=>t.groups.includes(feature)).map(t=>{
  const full=z?t.indices.reduce((v,i)=>v*z[i],t.coefficient):t.coefficient;
  const share=t.groups.filter(g=>g===feature).length/t.indices.length;
  return {...t,full,value:z?full*share:full};
 }).sort((a,b)=>Math.abs(b.value)-Math.abs(a.value));
 const note=document.createElement("p");note.style.fontSize="13px";
 note.textContent=z?"Ranked by absolute allocated contribution. Interaction terms are shared by coordinate multiplicity; all allocated terms sum to the feature bar.":"All nonzero complete-input terms containing this feature, ranked by absolute coefficient.";
 target.append(note);
 const max=Math.max(...rows.map(r=>Math.abs(r.value)),1e-15);
 for(const r of rows){
  const line=document.createElement("div");line.style.cssText="display:grid;grid-template-columns:minmax(160px,2fr) minmax(70px,1fr) 95px;gap:12px;align-items:center;margin:9px 0";
  const label=document.createElement("span");label.append(termMath(r));label.style.cssText="overflow-x:auto;min-width:0";
  const track=document.createElement("span");track.style.cssText="display:block;height:10px;border-radius:5px;width:"+100*Math.abs(r.value)/max+"%;background:"+(r.value>=0?"#9a63bf":"#009e78");
  const value=document.createElement("span");value.textContent=r.value.toPrecision(5);
  if(z){value.title="Full term contribution: "+r.full.toPrecision(7);
   const detail=document.createElement("small");detail.style.cssText="display:block;font-size:11px;color:#607487";
   detail.textContent="Full: "+r.full.toPrecision(5);value.append(detail)}
  line.append(label,track,value);target.append(line);
 }
 if(!rows.length){const p=document.createElement("p");p.textContent="No nonzero terms.";target.append(p)}
}
function openCoefficientDrill(element,feature){
 const root=element.closest(".equation-math"),a=JSON.parse(root.querySelector("[data-equation]").dataset.equation);
 renderTermDrill(root.querySelector(".coefficient-drill"),equationTerms(a,a.names,a.groups),feature);
}
