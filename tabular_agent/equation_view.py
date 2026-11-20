"""Compact equation previews with on-demand numerical expansion."""
import html
import json
from pathlib import Path
import numpy as np


def equation_preview(artifact, encoder, core):
    eqs=artifact["classes"]
    binary=len(eqs)==2
    last=eqs[1] if binary else eqs[-1]
    def coefficient(key):
        v=np.asarray(last[key],dtype=float)
        return v-np.asarray(eqs[0][key],dtype=float) if binary else v
    p=len(encoder["names"])
    taylor=artifact["kind"]=="native_taylor"
    if taylor:
        linear=coefficient("linear")
        h=coefficient("H")
        tensor=coefficient("T") if "T" in last else None
    else:
        a,b=coefficient("A"),coefficient("B")
    ranked=[]
    for group in dict.fromkeys(encoder["groups"]):
        cols=[j for j,g in enumerate(encoder["groups"]) if g==group]
        if taylor:
            ix=cols+[j+p for j in cols]
            norm=np.sum(linear[ix]**2)+.5*np.sum(h[ix,:]**2)
            if tensor is not None:
                norm+=np.sum(tensor[ix,:,:]**2)/6
        else:
            norm=np.sum(a[cols]**2+b[cols]**2)
        ranked.append((float(norm),group,cols))
    ranked.sort(key=lambda row:row[0],reverse=True)
    def variable(j):
        return "<msub><mi>z</mi><mtext>"+html.escape(encoder["names"][j])+"</mtext></msub>"
    rows=[]
    for _,group,cols in ranked[:5]:
        terms=[]
        for j in cols:
            values=[(float(linear[j]),1),(float(h[j,j]/2),2)] if taylor else [(float(b[j]),1),(float(a[j]),2)]
            if taylor and tensor is not None:
                values.append((float(tensor[j,j,j]/6),3))
            for value,power in reversed(values):
                if value==0:
                    continue
                v=variable(j)
                if power>1:
                    v=f"<msup>{v}<mn>{power}</mn></msup>"
                terms.append(("<mo>−</mo>" if value<0 else "<mo>+</mo>")+f"<mn>{abs(value):.5g}</mn>"+v)
        expr="".join(terms) or "<mn>0</mn>"
        if expr.startswith("<mo>+</mo>"):
            expr=expr[len("<mo>+</mo>"):]
        rows.append('<div style="padding:8px 12px;background:white;border-radius:8px;overflow-x:auto;min-width:0">'
                    '<strong style="font:13px system-ui">'+html.escape(group)+'</strong>'
                    '<math display="block" style="font-size:1.05em;width:max-content;margin:8px 0"><mrow>'+expr+'</mrow></math></div>')
    # Only coefficients needed for expansion are embedded, never kernels or rows.
    keys=("intercept","linear","H","T") if taylor else ("intercept","A","B","D")
    payload={"kind":artifact["kind"],"names":encoder["names"],"groups":encoder["groups"],
             "classes":[{k:e[k] for k in keys if k in e} for e in eqs]}
    handler=Path(__file__).with_name("equation_expand.js").read_text(encoding="utf-8")
    maximum=max((np.sqrt(row[0]) for row in ranked[:5]),default=1) or 1
    bars=[]
    for i,(norm,group,_) in enumerate(ranked[:5]):
        value=float(np.sqrt(norm))
        y=30+i*48
        click=html.escape("openCoefficientDrill(this,"+json.dumps(group)+")",quote=True)
        bars.append(f'<g tabindex="0" role="button" style="cursor:pointer" onclick="{click}" onkeydown="if(event.key===&quot;Enter&quot;||event.key===&quot; &quot;)this.click()">'
                    f'<text x="138" y="{y+5}" text-anchor="end" fill="#30495d" font-size="12">{html.escape(group)}</text>'
                    f'<rect x="150" y="{y-10}" width="{210*value/maximum:.2f}" height="22" rx="4" fill="#087ba4"/>'
                    f'<text x="{160+210*value/maximum:.2f}" y="{y+5}" fill="#536b7d" font-size="11">{value:.3g}</text></g>')
    chart=('<svg viewBox="0 0 425 280" role="img" aria-label="Top five features ranked by coefficient norm" '
           'style="width:100%;font-family:system-ui"><title>Top 5 coefficient-ranked features</title>'+
           "".join(bars)+'<text x="255" y="275" text-anchor="middle" font-size="11" fill="#607487">Coefficient norm</text></svg>')
    return ('<div class="equation-math">'+core+
            '<h3 style="font:600 14px system-ui;margin-top:20px">Top 5 coefficient-ranked features · main terms</h3>'
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;align-items:center">'
            '<div style="min-width:0">'+chart+'</div><div style="display:grid;gap:6px;min-width:0">'
            +"".join(rows)+'<div style="text-align:center;font-size:22px">…</div></div></div>'
            '<p style="font-size:13px">Click a ranked feature to inspect its terms, or choose a feature: <select onchange="openCoefficientDrill(this,this.value)"><option value="" disabled selected>Select a feature</option>'+
            "".join('<option>'+html.escape(g)+'</option>' for g in dict.fromkeys(encoder["groups"]))+'</select></p><div class="coefficient-drill"></div>'+
            '<script type="text/javascript">'+Path(__file__).with_name("term_drill.js").read_text(encoding="utf-8")+'</script>'+
            '<details style="margin-top:18px" data-equation="'+html.escape(json.dumps(payload,separators=(",",":")),quote=True)+
            '" ontoggle="'+html.escape(handler,quote=True)+'">'
            '<summary style="cursor:pointer;font:600 14px system-ui">Show full numerical equation</summary>'
            '<div class="full-equation" style="overflow-wrap:anywhere"></div></details></div>')
