/* Dependency-free loss charts. No predictive metrics or projections. */
window.TrainingView = {
 loss(canvas, rows, key, title, color) {
  const ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height;
  ctx.clearRect(0,0,w,h);ctx.font="13px system-ui";ctx.fillStyle="#536a7b";
  const points=rows.filter(r=>Number.isFinite(r[key]));
  if(!points.length){ctx.fillText(rows.length?"Not recorded in this archived run.":"Waiting for the first epoch…",60,50);return;}
  let lo=Math.min(...points.map(r=>r[key])),hi=Math.max(...points.map(r=>r[key]));
  if(lo===hi){lo-=.5;hi+=.5}
  const left=80,right=w-30,top=25,bottom=h-35,last=Math.max(points.at(-1).epoch,1);
  for(let i=0;i<=4;i++){
   const v=lo+(hi-lo)*i/4,y=bottom-(bottom-top)*i/4;
   ctx.strokeStyle="#e1e8ed";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();
   ctx.fillText(v.toPrecision(4),5,y+4);
  }
  ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();
  points.forEach((r,i)=>{const x=left+(r.epoch-1)/Math.max(last-1,1)*(right-left),y=bottom-(r[key]-lo)/(hi-lo)*(bottom-top);
   if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)});
  ctx.stroke();
  if(points.length===1){ctx.fillStyle=color;ctx.beginPath();ctx.arc(left,bottom-(points[0][key]-lo)/(hi-lo)*(bottom-top),3,0,Math.PI*2);ctx.fill();}
  ctx.fillStyle="#536a7b";ctx.fillText("Epoch 1",left,bottom+24);ctx.fillText("Epoch "+last,right-85,bottom+24);
  ctx.fillStyle=color;ctx.fillText(title+" · "+points.at(-1)[key].toFixed(4),left,16);
 }
};
