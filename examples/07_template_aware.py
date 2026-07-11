"""Template-aware prompt helper: lens the REAL chat-template-rendered tokens.
A/B system-prompt and thinking settings by how they steer an intended sense, check
whether a system message actually 'lands', and render a token-strip visualization."""
import os
from jlensvl import JLensVL, PromptHelper, viz

MID = os.environ.get("MODEL_DIR", "Qwen/Qwen3.5-4B")
LENS = os.environ.get("LENS", "lens.pt")
DEV = os.environ.get("DEV", "cuda:0")

jl = JLensVL.from_pretrained(MID, lens=LENS, device=DEV, multimodal=False)
ph = PromptHelper(jl)

senses = {"programming": ["programming", "language", "code", "software"],
          "island":      ["island", "islands", "province", "Indonesia"],
          "coffee":      ["coffee", "drink", "beverage", "espresso"]}
user   = {"role": "user",   "content": "Answer in one word. Java is a"}
sys_se = {"role": "system", "content": "You are a software engineering instructor."}
sys_geo= {"role": "system", "content": "You are a geography teacher about Indonesian islands."}

variants = {
    "no system":       {"messages": [user]},
    "sys=software eng": {"messages": [sys_se, user]},
    "sys=geography":    {"messages": [sys_geo, user]},
    "thinking on":      {"messages": [sys_se, user], "enable_thinking": True},
    "thinking off":     {"messages": [sys_se, user], "enable_thinking": False},
}

print("=== compare_templates (intended = programming) ===")
for r in ph.compare_templates([user], variants, senses, intended="programming"):
    print(f"  [{r['name']:16s}] programming={r['intended']:.2f}  "
          f"best_competitor={r['best_competitor']:.2f}  margin={r['margin']:+.2f}")

print("\n=== check_system_registers (software-eng system) ===")
print("  ", ph.check_system_registers([sys_se, user], senses, intended="programming"))

print("\n=== diagnose_thinking (does enabling <think> help or hurt 'programming'?) ===")
print("  ", ph.diagnose_thinking([sys_se, user], senses, intended="programming"))

tr = ph.trace_rendered([sys_se, user], senses=senses)
viz.rendered_strip_html(tr, intended="programming", out_path="template_strip.html",
                        title="template-aware J-Lens — Java (sys=software eng)")
print("\nwrote template_strip.html  |  rendered:", repr(tr["rendered"][:80]), "...")
