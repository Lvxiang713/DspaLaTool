"""Train at most once, then compare selected extractions using frozen kernels."""
import argparse
import hashlib
import html
import json
import re
from pathlib import Path
import shutil
import pandas as pd
from .workflow import run, dump
from .report import rendered_equation

CHOICES = {"geometric":("geometric",2), "taylor1":("taylor",1),
           "taylor2":("taylor",2), "taylor3":("taylor",3)}


def write_batch_pages(out, methods, rows, trained):
    table=pd.DataFrame(rows).drop(columns=["score_RMSE","score_R2"],errors="ignore")
    table.to_csv(out/"comparison.csv",index=False)
    names={"geometric":"Geometric additive","taylor1":"Taylor · first order",
           "taylor2":"Taylor · second order","taylor3":"Taylor · third order"}
    badges={"geometric":"Additive quadratic","taylor1":"Linear","taylor2":"Hessian","taylor3":"Cubic"}
    links="".join(f'<article><span class="badge">{badges[m]}</span><h3>{names[m]}</h3>'
                  f'<a class="primary" href="{m}/report.html" target="_blank">View analysis ↗</a>'
                  f'<a href="{m}/predictor.html" target="_blank">Predict ↗</a></article>' for m in methods)
    metric_names=[("ACC","ACC"),("MCC","MCC"),("F1_macro","F1"),("Precision_macro","Precision"),("Recall_macro","Recall")]
    tables=[]
    def fmt(value):
        return "—" if value is None else f"{value:.4f}"
    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        subset=[r for r in rows if r["dataset"]==dataset]
        first=subset[0]
        body='<tr class="original"><td>Original model</td>'+"".join(
            "<td>"+fmt(first.get("original_"+k))+"</td>" for k,_ in metric_names)+"<td>1.0000</td></tr>"
        for row in subset:
            body+="<tr><td>"+names[row["method"]]+"</td>"+"".join(
                "<td>"+fmt(row.get("quadratic_"+k))+"</td>" for k,_ in metric_names)+(
                "<td>"+fmt(row.get("prediction_agreement"))+"</td></tr>")
        tables.append('<section><header><h2>'+html.escape(dataset.replace("_"," ").title())+
            '</h2><span class="badge">'+str(first["n"])+' samples</span></header>'
            '<div class="table-wrap"><table><thead><tr><th>Model / expression</th>'+
            "".join("<th>"+label+"</th>" for _,label in metric_names)+
            '<th>Agreement</th></tr></thead><tbody>'+body+'</tbody></table></div></section>')
    page=('<!doctype html><html lang="en"><meta charset="utf-8"><title>Equation comparison</title>'
          '<meta name="viewport" content="width=device-width"><style>'
          'body{font:14px system-ui;margin:0;padding:24px;color:#173248;background:#f7f9fc}'
          'h1{font-size:26px;margin:0}h2{font-size:18px}h3{font-size:16px;margin:12px 0 20px}'
          '.top,header{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}'
          '.methods{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:22px 0}'
          'article,section{background:white;border:1px solid #dfe8ef;border-radius:12px;padding:18px}'
          'article a{display:inline-block;margin-right:12px;text-decoration:none;font-size:13px}'
          'a{color:#087ba4}.primary{background:#087ba4;color:white;padding:9px 12px;border-radius:6px}'
          '.badge{background:#edf3f8;color:#536b7d;border-radius:20px;padding:5px 10px;font-size:12px}'
          'section{margin-top:18px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}'
          'td,th{padding:13px 10px;border-bottom:1px solid #e8edf2;text-align:right;white-space:nowrap}'
          'td:first-child,th:first-child{text-align:left}th{color:#607487;font-weight:500}'
          '.original{background:#f3f6fa}tbody tr:last-child td{border-bottom:0}</style>'
          '<div class="top"><h1>Equation comparison</h1><a class="primary" href="comparison.csv">Download CSV</a></div>'
          '<div class="methods">'+links+'</div>'+"".join(tables)+'</html>')
    (out/"report.html").write_text(page,encoding="utf-8")
    # A single standalone predictor with method tabs, rather than nested iframes.
    payload={}
    for method in methods:
        directory=out/method
        artifact=json.loads((directory/"equation.json").read_text(encoding="utf-8"))
        summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
        encoder=json.loads((directory/"encoder.json").read_text(encoding="utf-8"))
        payload[method]={"equation":artifact,"encoder":encoder,
            "labels":summary["classes"],"formula":rendered_equation(artifact,encoder)}
    tabs='<nav class="method-tabs">'+"".join(
        f'<button class="method-tab{" active" if i==0 else ""}" data-method="{m}">{names[m]}</button>'
        for i,m in enumerate(methods))+'</nav>'
    predictor=(out/methods[0]/"predictor.html").read_text(encoding="utf-8")
    predictor=predictor.replace("const DATA=","let DATA=",1)
    predictor=re.sub(r"<h1>.*?</h1>","<h1>Equation predictor</h1>"+tabs,predictor,count=1)
    predictor=predictor.replace("</style>",
        '.method-tabs{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0}.method-tab{background:white;color:#486378;border-color:#d5e1e9}'
        '.method-tab.active{background:#087ba4;color:white;border-color:#087ba4}</style>',1)
    switch=('<script>const METHODS='+json.dumps(payload,allow_nan=False).replace("<","\\u003c")+
            ';document.querySelectorAll(".method-tab").forEach(b=>b.onclick=()=>{DATA=METHODS[b.dataset.method];'
            'document.querySelector(".equation-math").outerHTML=DATA.formula;'
            'document.querySelectorAll(".method-tab").forEach(t=>t.classList.toggle("active",t===b));'
            'document.querySelector("#result").textContent="";document.querySelector("#contributions").replaceChildren();'
            'document.querySelector("#inputError").textContent="";document.querySelector("#error").textContent="";});</script>')
    (out/"predictor.html").write_text(predictor.replace("</html>",switch+"</html>"),encoding="utf-8")


def batch(csv, target, output, config, methods, external=None, parent=None):
    if not methods or len(set(methods)) != len(methods) or any(m not in CHOICES for m in methods):
        raise ValueError("Select one or more unique extraction methods.")
    out = Path(output)
    out.mkdir(parents=True,exist_ok=False)
    dump(out/"status.json",{"stage":"analysis" if parent else "training","status":"running"})
    try:
        rows = []
        frozen = parent
        for method in methods:
            cfg = dict(config,approximation=CHOICES[method][0],approximation_order=CHOICES[method][1])
            cfg["requested_methods"] = methods
            cfg.pop("monitor_sink",None)
            if not frozen:
                cfg["monitor_sink"] = str((out/"monitor.jsonl").resolve())
            if frozen:
                cfg["reuse_analysis"] = str(frozen)
            dump(out/"status.json",{"stage":f"analyzing_{method}" if frozen else "training",
                                   "status":"running","method":method,"training_performed":not bool(parent)})
            child = out/method
            summary = run(csv,target,child,cfg,external)
            if not frozen:
                frozen = child.resolve()
                if (child/"monitor.jsonl").exists():
                    shutil.copyfile(child/"monitor.jsonl",out/"monitor.jsonl")
            for dataset,result in summary["results"].items():
                row = {"method":method,"dataset":dataset,"n":result["n"],
                       "prediction_agreement":result.get("prediction_agreement")}
                row.update({f"{kind}_{k}":v for kind in ("original","quadratic")
                            for k,v in (result.get(kind) or {}).items()})
                rows.append(row)
        table = pd.DataFrame(rows)
        write_batch_pages(out,methods,rows,not bool(parent))
        dump(out/"summary.json",{"methods":methods,"training_performed":not bool(parent),
                                "frozen_parent":str(frozen),"comparisons":json.loads(table.to_json(orient="records"))})
        dump(out/"ARTIFACT_MANIFEST.json",{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
             for p in out.iterdir() if p.is_file() and p.name not in ("status.json","ARTIFACT_MANIFEST.json")})
        dump(out/"status.json",{"stage":"complete","status":"complete"})
    except Exception as exc:
        dump(out/"status.json",{"stage":"failed","status":"error","message":str(exc)})
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("request")
    obj = json.loads(Path(p.parse_args().request).read_text(encoding="utf-8"))
    batch(**obj)


if __name__ == "__main__":
    main()
