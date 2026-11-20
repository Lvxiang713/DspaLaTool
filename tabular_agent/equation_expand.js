if(this.open&&!this.dataset.rendered){
 const d=JSON.parse(this.dataset.equation),ns="http://www.w3.org/1998/Math/MathML";
 const node=(tag,text)=>{const e=document.createElementNS(ns,tag);if(text!==undefined)e.textContent=text;return e};
 const sub=(symbol,name)=>{const e=node("msub");e.append(node("mi",symbol),node("mtext",name));return e};
 const binary=d.classes.length===2,eqs=binary?[d.classes[1]]:d.classes;
 const base=binary?d.classes[0]:null,p=d.names.length;
 const target=this.querySelector(".full-equation");
 const coord=i=>sub("z",d.names[i]);
 const product=indices=>{
  const counts=new Map();
  for(const i of indices)counts.set(i,(counts.get(i)||0)+1);
  return [...counts].map(([i,power])=>{
   if(power===1)return coord(i);
   const term=node("msup");term.append(coord(i),node("mn",String(power)));return term;
  });
 };
 eqs.forEach((e,c)=>{
  const block=document.createElement("div");block.style.cssText="margin:16px 0;max-height:520px;overflow:auto;background:white;border-radius:8px;padding:16px";
  const line=(parts)=>{
   const math=node("math"),row=node("mrow");
   math.style.cssText="font-size:1em;display:block;width:max-content;margin:0";
   row.append(...parts);math.append(row);
   const wrapper=document.createElement("div");wrapper.style.cssText="padding:7px 0";
   wrapper.append(math);block.append(wrapper);
  };
  const label=binary?node("mi","Q"):sub("S",String(c));
  line([label,node("mo","="),node("mn",Number((e.intercept-(base?base.intercept:0)).toPrecision(7)).toString())]);
  const add=(v,factors)=>{
   if(v===0)return;
   line([node("mo",v<0?"−":"+"),node("mn",Number(Math.abs(v).toPrecision(7)).toString()),...factors]);
  };
  if(d.kind==="native_taylor"){
   const n=p;
   for(let i=0;i<n;i++)add(e.linear[i]-(base?base.linear[i]:0),[coord(i)]);
   for(let i=0;i<n;i++)for(let j=i;j<n;j++)
    add((e.H[i][j]-(base?base.H[i][j]:0))*(i===j?.5:1),product([i,j]));
   if(e.T)for(let i=0;i<n;i++)for(let j=i;j<n;j++)for(let k=j;k<n;k++){
    const factor=i===k?1/6:(i===j||j===k?.5:1);
    add((e.T[i][j][k]-(base?base.T[i][j][k]:0))*factor,product([i,j,k]));
   }
  }else{
   for(let j=0;j<p;j++){
    add(e.A[j]-(base?base.A[j]:0),product([j,j]));
    add(e.B[j]-(base?base.B[j]:0),[sub("z",d.names[j])]);
   }
  }
  target.append(block);
 });
 this.dataset.rendered="true";
}
