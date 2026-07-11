"""Self-contained HTML visualizations for the J-Lens.

- `slice_grid_html`: the flagship layer x position "slice grid" — for every token
  position and every layer, the concept the model is poised to say, as a colored,
  hoverable grid (top-1 in the cell, full top-k on hover). Shows a concept forming
  as it climbs the layers, and the answer resolving at the top.
- `race_chart_html`: an inline-SVG line chart of competing concepts across layers
  (e.g. contradictory image+text) — watch one override the other.

Both return a self-contained HTML string (inline CSS/JS, no external deps) and
optionally write it to `out_path`. Works offline; open in any browser.
"""
from __future__ import annotations
import html as _h
import json

_HEAD = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>
:root{{--bg:#0d1117;--pan:#161b22;--bd:#30363d;--tx:#e6edf3;--mu:#8b949e;--ac:#79c0ff;--gn:#7ee787;--or:#ffa657}}
@media(prefers-color-scheme:light){{:root{{--bg:#fff;--pan:#f6f8fa;--bd:#d0d7de;--tx:#1f2328;--mu:#656d76;--ac:#0969da;--gn:#1a7f37;--or:#bc4c00}}}}
body{{margin:0;background:var(--bg);color:var(--tx);font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:18px}}
h1{{font-size:1.2em;margin:.2em 0}}.sub{{color:var(--mu);margin:.2em 0 1em;font-size:.9em}}
.wrap{{overflow:auto;border:1px solid var(--bd);border-radius:8px;max-height:82vh}}
table{{border-collapse:collapse;font:12px/1.2 ui-monospace,Menlo,monospace}}
th,td{{border:1px solid var(--bd);padding:3px 6px;white-space:nowrap;text-align:center}}
thead th{{position:sticky;top:0;background:var(--pan);color:var(--ac);z-index:2}}
th.ly{{position:sticky;left:0;background:var(--pan);color:var(--mu);z-index:1}}
td{{cursor:default;color:var(--tx)}}td:hover{{outline:2px solid var(--ac)}}
.mo{{font-weight:700}}.legend{{color:var(--mu);font-size:.85em;margin:.6em 0}}
.spark-tip{{position:fixed;z-index:9999;pointer-events:none;display:none;background:var(--pan);border:1px solid var(--bd);border-radius:6px;padding:6px 8px;box-shadow:0 4px 16px rgba(0,0,0,.4);font:11px/1.35 ui-monospace,Menlo,monospace;color:var(--tx);max-width:240px}}
.spark-tip .st-h{{color:var(--ac);font-weight:700;margin-bottom:3px}}
.spark-tip .st-k{{color:var(--mu);margin-top:3px;white-space:normal;word-break:break-word}}
.spark-tip svg{{display:block}}
</style></head><body>"""


def _grid(jl, ll, ml, ids, layers, topk):
    tokens = [(jl.tok.decode([int(i)]).replace("\n", "\\n").strip() or "·") for i in ids]
    seq = len(ids)
    cell, vmax = {}, 1e-6
    for L in layers:
        M = ll[L]
        for p in range(seq):
            v, idx = M[p].topk(topk)
            top = [(jl.tok.decode([int(i)]).replace("\n", "\\n").strip() or "·") for i in idx.tolist()]
            s = float(v[0]); vmax = max(vmax, s)
            cell[(L, p)] = (top, s)
    mrow = {}
    for p in range(seq):
        idx = ml[p].topk(topk).indices
        mrow[p] = [(jl.tok.decode([int(i)]).replace("\n", "\\n").strip() or "·") for i in idx.tolist()]
    return tokens, cell, mrow, seq, vmax


def slice_grid_html(jl, prompt, *, layers=None, topk=5, out_path=None, title="JLensVL slice grid"):
    """Layer x position J-Lens slice grid for a text `prompt`."""
    jl._require_lens()
    layers = sorted(layers if layers is not None else jl.lens.source_layers)
    ll, ml, ids = jl.lens.apply(jl.lm, prompt, positions=None, layers=layers, use_jacobian=True)
    if hasattr(ids, "dim") and ids.dim() > 1:
        ids = ids[0]
    tokens, cell, mrow, seq, vmax = _grid(jl, ll, ml, ids, layers, topk)

    def color(s):
        a = max(0.0, min(1.0, s / vmax))
        return f"background:rgba(126,231,135,{a*0.8:.2f})"

    # Per-position top-1 trajectory across depth (layers ascending) for the sparkline.
    asc = sorted(layers)
    traj = {p: [round(cell[(L, p)][1], 4) for L in asc] for p in range(seq)}
    import json as _json
    traj_js = _json.dumps({str(p): traj[p] for p in range(seq)}, separators=(",", ":"))
    layers_js = _json.dumps(asc, separators=(",", ":"))
    toks_js = _json.dumps([tokens[p] for p in range(seq)], separators=(",", ":"))

    head = "".join(f'<th data-p="{p}" title="position {p}">{_h.escape(tokens[p])}</th>' for p in range(seq))
    rows = ""
    for L in reversed(layers):                       # deep layers on top
        cells = ""
        for p in range(seq):
            top, s = cell[(L, p)]
            cells += (f'<td data-p="{p}" style="{color(s)}" '
                      f'title="{_h.escape(" · ".join(top))}">{_h.escape(top[0])}</td>')
        rows += f'<tr><th class="ly">L{L}</th>{cells}</tr>'
    mcells = "".join(f'<td class="mo" data-p="{p}" title="{_h.escape(" · ".join(mrow[p]))}">{_h.escape(mrow[p][0])}</td>' for p in range(seq))
    rows += f'<tr><th class="ly">OUT</th>{mcells}</tr>'

    script = (
        "<script>(function(){"
        f"const TRAJ={traj_js},LAYERS={layers_js},TOKS={toks_js};"
        "const tip=document.createElement('div');tip.className='spark-tip';"
        "document.body.appendChild(tip);"
        "function spark(a){"
        "const W=120,H=36,pad=3;"
        "if(!a||!a.length)return '';"
        "let mn=Math.min.apply(null,a),mx=Math.max.apply(null,a);"
        "const rng=(mx-mn)||1;"
        "const n=a.length;"
        "const xs=i=>pad+(n<2?0:i/(n-1)*(W-2*pad));"
        "const ys=v=>H-pad-(v-mn)/rng*(H-2*pad);"
        "let pts=a.map((v,i)=>xs(i).toFixed(1)+','+ys(v).toFixed(1)).join(' ');"
        "let lx=xs(n-1).toFixed(1),ly=ys(a[n-1]).toFixed(1);"
        "return '<svg width=\"'+W+'\" height=\"'+H+'\" viewBox=\"0 0 '+W+' '+H+'\">'"
        "+'<polyline fill=\"none\" stroke=\"#7ee787\" stroke-width=\"1.6\" points=\"'+pts+'\"/>'"
        "+'<circle cx=\"'+lx+'\" cy=\"'+ly+'\" r=\"2.2\" fill=\"#79c0ff\"/></svg>';"
        "}"
        "function show(p,x,y){"
        "const a=TRAJ[String(p)];if(!a)return;"
        "const tk=TOKS[p]||'',lo=LAYERS[0],hi=LAYERS[LAYERS.length-1];"
        "tip.innerHTML='<div class=\"st-h\">pos '+p+' · '+esc(tk)+'</div>'"
        "+spark(a)"
        "+'<div class=\"st-k\">top-1 score L'+lo+'→L'+hi+' · '+a[0].toFixed(2)+' → '+a[a.length-1].toFixed(2)+'</div>';"
        "tip.style.display='block';move(x,y);"
        "}"
        "function esc(s){return String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));}"
        "function move(x,y){"
        "const w=tip.offsetWidth,h=tip.offsetHeight;"
        "let nx=x+14,ny=y+14;"
        "if(nx+w>window.innerWidth)nx=x-w-14;"
        "if(ny+h>window.innerHeight)ny=y-h-14;"
        "tip.style.left=Math.max(2,nx)+'px';tip.style.top=Math.max(2,ny)+'px';"
        "}"
        "function hide(){tip.style.display='none';}"
        "const tbl=document.currentScript.previousElementSibling.querySelector('table');"
        "tbl.addEventListener('mouseover',e=>{"
        "const c=e.target.closest('[data-p]');if(!c)return;show(+c.getAttribute('data-p'),e.clientX,e.clientY);"
        "});"
        "tbl.addEventListener('mousemove',e=>{if(tip.style.display==='block')move(e.clientX,e.clientY);});"
        "tbl.addEventListener('mouseout',e=>{if(!e.relatedTarget||!e.relatedTarget.closest('[data-p]'))hide();});"
        "})();</script>"
    )

    doc = (_HEAD.format(title=_h.escape(title)) +
           f'<h1>{_h.escape(title)}</h1>'
           f'<p class="sub">J-Lens 每层每位置「欲言」的概念 · 越绿=越确定 · hover 看 top-{topk} 与该位置 top-1 分数逐层轨迹 · 深层在上，OUT=模型真实输出</p>'
           f'<div class="wrap"><table><thead><tr><th class="ly">layer</th>{head}</tr></thead><tbody>{rows}</tbody></table></div>'
           f'{script}'
           f'<p class="legend">prompt: {_h.escape(prompt)}</p></body></html>')
    if out_path:
        open(out_path, "w").write(doc)
    return doc


def slice_grid_image_html(jl, image, question, *, layers=None, topk=5, out_path=None,
                          title="JLensVL VLM slice grid", max_text_positions=None):
    """Layer x position slice grid for a VLM (image + question). Image-token
    positions are collapsed to a single '[IMG]' column band to keep it legible;
    the interesting text positions (post-image + answer) get full columns."""
    jl._require_lens()
    from jlensvl.core import JLensVL  # noqa
    from jlens.hooks import ActivationRecorder
    import torch
    inputs = jl._vlm_inputs(image, question)
    ids = inputs["input_ids"][0]
    img_pos = set((ids == jl.image_token_id).nonzero(as_tuple=True)[0].tolist()) if jl.image_token_id else set()
    layers = sorted(layers if layers is not None else jl.lens.source_layers)
    with torch.no_grad(), ActivationRecorder(jl.lm.layers, at=layers) as rec:
        jl.model(**inputs)
        acts = {i: rec.activations[i].detach() for i in layers}
    # columns: one [IMG] band + each non-image position
    text_pos = [p for p in range(len(ids)) if p not in img_pos]
    cols = ([("img", None)] if img_pos else []) + [("txt", p) for p in text_pos]
    tok = lambda i: (jl.tok.decode([int(i)]).replace("\n", "\\n").strip() or "·")

    def readout(L, pos, k):
        r = acts[L][0, pos].float()
        lg = jl.lm.unembed(jl.lens.transport(r, L)[None].to(r.device))[0]
        v = lg.topk(k)
        return [tok(i) for i in v.indices.tolist()], float(v.values[0])

    img_mid = sorted(img_pos)[len(img_pos)//2] if img_pos else None
    # per-column labels ([IMG] band or decoded token) — the TOKS-equivalent array
    col_labels = ["[IMG]" if kind == "img" else tok(ids[p]) for kind, p in cols]
    header = "".join(f'<th data-p="{j}">{_h.escape(col_labels[j])}</th>' for j in range(len(cols)))
    vmax = 1e-6; grid = {}
    for L in layers:
        for j, (kind, p) in enumerate(cols):
            pos = img_mid if kind == "img" else p
            top, s = readout(L, pos, topk); vmax = max(vmax, s); grid[(L, j)] = (top, s)
    rows = ""
    for L in reversed(layers):
        cells = ""
        for j in range(len(cols)):
            top, s = grid[(L, j)]
            a = max(0.0, min(1.0, s / vmax))
            cells += f'<td data-p="{j}" style="background:rgba(126,231,135,{a*0.8:.2f})" title="{_h.escape(" · ".join(top))}">{_h.escape(top[0])}</td>'
        rows += f'<tr><th class="ly">L{L}</th>{cells}</tr>'

    # Per-column top-1 trajectory across depth (layers ascending) for the sparkline.
    asc = sorted(layers)
    traj = {j: [round(grid[(L, j)][1], 4) for L in asc] for j in range(len(cols))}
    traj_js = json.dumps({str(j): traj[j] for j in range(len(cols))}, separators=(",", ":"))
    layers_js = json.dumps(asc, separators=(",", ":"))
    toks_js = json.dumps(col_labels, separators=(",", ":"))

    script = (
        "<script>(function(){"
        f"const TRAJ={traj_js},LAYERS={layers_js},TOKS={toks_js};"
        "const tip=document.createElement('div');tip.className='spark-tip';"
        "document.body.appendChild(tip);"
        "function spark(a){"
        "const W=120,H=36,pad=3;"
        "if(!a||!a.length)return '';"
        "let mn=Math.min.apply(null,a),mx=Math.max.apply(null,a);"
        "const rng=(mx-mn)||1;"
        "const n=a.length;"
        "const xs=i=>pad+(n<2?0:i/(n-1)*(W-2*pad));"
        "const ys=v=>H-pad-(v-mn)/rng*(H-2*pad);"
        "let pts=a.map((v,i)=>xs(i).toFixed(1)+','+ys(v).toFixed(1)).join(' ');"
        "let lx=xs(n-1).toFixed(1),ly=ys(a[n-1]).toFixed(1);"
        "return '<svg width=\"'+W+'\" height=\"'+H+'\" viewBox=\"0 0 '+W+' '+H+'\">'"
        "+'<polyline fill=\"none\" stroke=\"#7ee787\" stroke-width=\"1.6\" points=\"'+pts+'\"/>'"
        "+'<circle cx=\"'+lx+'\" cy=\"'+ly+'\" r=\"2.2\" fill=\"#79c0ff\"/></svg>';"
        "}"
        "function show(p,x,y){"
        "const a=TRAJ[String(p)];if(!a)return;"
        "const tk=TOKS[p]||'',lo=LAYERS[0],hi=LAYERS[LAYERS.length-1];"
        "tip.innerHTML='<div class=\"st-h\">pos '+p+' · '+esc(tk)+'</div>'"
        "+spark(a)"
        "+'<div class=\"st-k\">top-1 score L'+lo+'→L'+hi+' · '+a[0].toFixed(2)+' → '+a[a.length-1].toFixed(2)+'</div>';"
        "tip.style.display='block';move(x,y);"
        "}"
        "function esc(s){return String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));}"
        "function move(x,y){"
        "const w=tip.offsetWidth,h=tip.offsetHeight;"
        "let nx=x+14,ny=y+14;"
        "if(nx+w>window.innerWidth)nx=x-w-14;"
        "if(ny+h>window.innerHeight)ny=y-h-14;"
        "tip.style.left=Math.max(2,nx)+'px';tip.style.top=Math.max(2,ny)+'px';"
        "}"
        "function hide(){tip.style.display='none';}"
        "const tbl=document.currentScript.previousElementSibling.querySelector('table');"
        "tbl.addEventListener('mouseover',e=>{"
        "const c=e.target.closest('[data-p]');if(!c)return;show(+c.getAttribute('data-p'),e.clientX,e.clientY);"
        "});"
        "tbl.addEventListener('mousemove',e=>{if(tip.style.display==='block')move(e.clientX,e.clientY);});"
        "tbl.addEventListener('mouseout',e=>{if(!e.relatedTarget||!e.relatedTarget.closest('[data-p]'))hide();});"
        "})();</script>"
    )

    doc = (_HEAD.format(title=_h.escape(title)) +
           f'<h1>{_h.escape(title)}</h1>'
           f'<p class="sub">VLM J-Lens · 图像 token 折叠成 [IMG] 单列 · 关注 post-image 与答案位视觉概念在哪 verbalize · hover 看 top-{topk} 与该列 top-1 分数逐层轨迹</p>'
           f'<div class="wrap"><table><thead><tr><th class="ly">layer</th>{header}</tr></thead><tbody>{rows}</tbody></table></div>'
           f'{script}'
           f'<p class="legend">image: {_h.escape(str(image))} · question: {_h.escape(question)}</p></body></html>')
    if out_path:
        open(out_path, "w").write(doc)
    return doc


def rendered_strip_html(trace, *, intended=None, out_path=None, title="template-aware J-Lens"):
    """Token strip over the chat-template-rendered sequence. Special/template tokens
    are highlighted (the invisible layer between your text and the model); hover any
    token to see the concept the model is poised to say there. `trace` = output of
    PromptHelper.trace_rendered."""
    boxes = ""
    for p, c in enumerate(trace["per"]):
        surf = c["tok"].replace("\n", "\\n") or "·"
        tip = _h.escape("L%d 欲言: %s" % (trace["layer"], " · ".join(c["top"])))
        cls = "sp" if c["special"] else "ct"
        if p == trace["answer"]:
            cls += " ans"
        role = c.get("role")
        rl = f' data-role="{role}"' if role else ""
        boxes += f'<span class="tk {cls}"{rl} title="{tip}">{_h.escape(surf)}</span>'
    senses_html = ""
    if trace.get("senses"):
        vmax = max(trace["senses"].values()) or 1e-6
        rows = ""
        for n, v in sorted(trace["senses"].items(), key=lambda x: -x[1]):
            w = max(2, int(v / vmax * 240))
            hl = "background:var(--gn)" if n == intended else "background:var(--mu)"
            tag = " ← intended" if n == intended else ""
            rows += (f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
                     f'<span style="width:120px;color:var(--mu)">{_h.escape(n)}{tag}</span>'
                     f'<span style="height:12px;width:{w}px;border-radius:3px;{hl}"></span>'
                     f'<span>{v:.2f}</span></div>')
        senses_html = (f'<h3 style="margin:1.2em 0 .3em;color:var(--ac)">答案位各义分数 (L{trace["layer"]})</h3>{rows}')
    css = ('.tk{display:inline-block;padding:2px 4px;margin:1px;border-radius:4px;'
           'font:12px/1.4 ui-monospace,Menlo,monospace;border:1px solid transparent}'
           '.ct{background:rgba(126,231,135,.14)}'
           '.sp{background:rgba(121,192,255,.22);color:var(--ac);border-color:var(--bd)}'
           '.ans{outline:2px solid var(--or);font-weight:700}'
           '.tk:hover{border-color:var(--ac)}')
    doc = (_HEAD.format(title=_h.escape(title)).replace("</style>", css + "</style>") +
           f'<h1>{_h.escape(title)}</h1>'
           f'<p class="sub">在<b>真实模板渲染的 token</b> 上跑 J-Lens（不是裸字符串）。'
           f'<span class="sp" style="padding:1px 4px">蓝色</span>=模板注入的特殊 token（你看不见的那层）· '
           f'<span class="ans" style="padding:1px 4px">橙框</span>=答案位 · hover 看该位置「欲言」的概念。</p>'
           f'<div style="line-height:2.2">{boxes}</div>{senses_html}</body></html>')
    if out_path:
        open(out_path, "w").write(doc)
    return doc


def race_chart_html(race, concept_a, concept_b, *, out_path=None, title="concept race",
                    crossover=None):
    """Inline-SVG line chart from a `concept_race` result {layer: {name: score}}."""
    layers = sorted(race.keys())
    A = [race[L][concept_a] for L in layers]
    B = [race[L][concept_b] for L in layers]
    W, H, pad = 720, 320, 46
    ymax = max(max(A), max(B)) * 1.08
    X = lambda l: pad + (l - layers[0]) / (layers[-1] - layers[0]) * (W - 2 * pad)
    Y = lambda v: H - pad - v / ymax * (H - 2 * pad)
    if crossover is None:
        for L in layers:
            if race[L][concept_a] > race[L][concept_b]:
                crossover = L; break

    def poly(vals, col):
        pts = " ".join(f"{X(l):.1f},{Y(v):.1f}" for l, v in zip(layers, vals))
        return f'<polyline fill="none" stroke="{col}" stroke-width="2.4" points="{pts}"/>'
    grid = "".join(f'<line x1="{pad}" y1="{Y(g):.1f}" x2="{W-pad}" y2="{Y(g):.1f}" stroke="var(--bd)"/>'
                   f'<text x="{pad-8}" y="{Y(g)+4:.1f}" text-anchor="end" font-size="11" fill="var(--mu)">{g}</text>'
                   for g in range(0, int(ymax), 10))
    xlab = "".join(f'<text x="{X(l):.1f}" y="{H-pad+18}" text-anchor="middle" font-size="10" fill="var(--mu)">L{l}</text>'
                   for l in layers if l % 3 == 0)
    cx = X(crossover) if crossover else None
    cl = (f'<line x1="{cx:.1f}" y1="{pad}" x2="{cx:.1f}" y2="{H-pad}" stroke="var(--or)" stroke-dasharray="4 3"/>'
          f'<text x="{cx:.1f}" y="{pad-6}" text-anchor="middle" font-size="11" fill="var(--or)">L{crossover} 反超</text>') if cx else ""
    svg = (f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px">{grid}{cl}'
           f'{poly(B, "var(--or)")}{poly(A, "var(--gn)")}'
           f'<text x="{W-pad}" y="{pad}" text-anchor="end" font-size="12" fill="var(--gn)">● {_h.escape(concept_a)}</text>'
           f'<text x="{W-pad}" y="{pad+18}" text-anchor="end" font-size="12" fill="var(--or)">● {_h.escape(concept_b)}</text></svg>')
    doc = (_HEAD.format(title=_h.escape(title)) + f'<h1>{_h.escape(title)}</h1>'
           f'<p class="sub">两个概念在 J-Lens 里逐层竞争 · 交叉点 = 一方压过另一方</p>{svg}</body></html>')
    if out_path:
        open(out_path, "w").write(doc)
    return doc
