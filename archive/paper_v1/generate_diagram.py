"""Generate HopField-Mamba architecture diagram as .drawio XML, then export to PNG."""

import xml.etree.ElementTree as ET
import subprocess, os, tempfile, shutil

OUT_DIR = "/home/jupyter-238w1a5447/genomics/Genomic-FM/research_paper"
W = 1600  # canvas width
H = 900   # canvas height (16:9)

# ── Helper to build a cell ──────────────────────────────────────────
def cell(id_, value, style, x, y, w, h, parent="1"):
    c = ET.SubElement(root, "mxCell", id=id_, value=value, style=style,
                      vertex="1", parent=parent)
    geo = ET.SubElement(c, "mxGeometry", x=str(x), y=str(y),
                        width=str(w), height=str(h), as_="geometry")
    return c

def edge(id_, source, target, label="", style=""):
    attrs = {"id": id_, "edge": "1", "source": source, "target": target, "parent": "1"}
    if style:
        attrs["style"] = style
    e = ET.SubElement(root, "mxCell", **attrs)
    if label:
        v = ET.SubElement(e, "mxGeometry", x="0", y="0", as_="geometry")
        v2 = ET.SubElement(e, "mxGeometry", relative="1", as_="geometry")
        ET.SubElement(e, "mxCell", id=id_ + "_label", value=label,
                      style="text;html=1;strokeColor=none;fillColor=none;align=center;"
                            "verticalAlign=middle;fontSize=11;fontFamily=Helvetica;",
                      vertex="1", connectable="0")
    return e

# ── Build XML ──────────────────────────────────────────────────────
mxGraphModel = ET.Element("mxGraphModel", dx="0", dy="0", grid="1",
                          gridSize="10", guides="1", tooltips="1",
                          connect="1", arrows="1", fold="1", page="1",
                          pageScale="1", pageWidth=str(W), pageHeight=str(H),
                          background="none", math="0", shadow="0")
root = ET.SubElement(mxGraphModel, "root")

ET.SubElement(root, "mxCell", id="0")
ET.SubElement(root, "mxCell", id="1", parent="0")

# ══════════════════════════════════════════════════════════════════════
# COLOR PALETTE
# ══════════════════════════════════════════════════════════════════════
NAVY_DARK   = "#1B2A4A"
NAVY_MED    = "#2C4A7C"
NAVY_LIGHT  = "#3D6AA8"
NAVY_PALE   = "#D6E4F0"
TEAL        = "#2A9D8F"
ORANGE      = "#E76F51"
ORANGE_LIGHT= "#F4A261"
RED_ACCENT  = "#E63946"
GRAY_LINE   = "#B0B8C1"
GRAY_TEXT   = "#6B7280"

# Positions
MID_Y = H // 2  # 450
L1_Y  = 150     # Panel (a) pipeline
L2_Y  = 550     # Panel (b) layer detail

# ── PANEL (a): Overall Pipeline ────────────────────────────────────
PIPELINE_W = 140
PIPELINE_H = 60
GAP = 40
START_X = 160

boxes_a = [
    ("DNA\nSequence",       NAVY_DARK),
    ("Token\nEmbed",        NAVY_MED),
    ("HFM\n×6",             NAVY_LIGHT),
    ("Decoder\n×2",         NAVY_MED),
    ("Output\nPrediction",  NAVY_DARK),
]

box_ids_a = []
for i, (label, color) in enumerate(boxes_a):
    x = START_X + i * (PIPELINE_W + GAP)
    y = L1_Y - PIPELINE_H // 2
    cid = f"a{i}"
    box_ids_a.append(cid)
    fill = color
    font_color = "#FFFFFF"
    cell(cid, label,
         f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};"
         f"fontColor={font_color};fontSize=13;fontStyle=1;"
         f"fontFamily=Helvetica;arcSize=12;strokeColor={fill};"
         f"shadow=1;",
         x, y, PIPELINE_W, PIPELINE_H)

# Arrows between pipeline boxes
for i in range(len(box_ids_a) - 1):
    sid = box_ids_a[i]
    tid = box_ids_a[i + 1]
    lbl = "{A,C,G,T}" if i == 0 else ""
    edge(f"ae{i}", sid, tid, lbl,
         "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
         "jettySize=auto;html=1;strokeWidth=2;strokeColor=" + GRAY_LINE + ";"
         "endArrow=blockThin;endSize=12;")

# Brace above pipeline
center_x = START_X + (len(boxes_a) * (PIPELINE_W + GAP) - GAP) // 2
brace_x0 = START_X - 10
brace_x1 = START_X + len(boxes_a) * (PIPELINE_W + GAP) - GAP + 10
brace_y = L1_Y - PIPELINE_H // 2 - 30

cell("abrace", "",
     "shape=curlyBracket;whiteSpace=wrap;html=1;rounded=1;"
     "strokeColor=" + NAVY_MED + ";strokeWidth=2;",
     brace_x0, brace_y, brace_x1 - brace_x0, 20)
cell("abracelabel", "Masked Autoencoder (MAE)",
     "text;html=1;strokeColor=none;fillColor=none;align=center;"
     "verticalAlign=middle;fontSize=14;fontStyle=1;fontFamily=Helvetica;"
     "fontColor=" + NAVY_MED + ";",
     brace_x0, brace_y - 30, brace_x1 - brace_x0, 25)

# Panel (a) label
cell("alabel", "Part (a) — Overall HopField-Mamba MAE Pipeline",
     "text;html=1;strokeColor=none;fillColor=none;align=center;"
     "verticalAlign=middle;fontSize=15;fontStyle=1;fontFamily=Helvetica;"
     "fontColor=" + GRAY_TEXT + ";",
     0, L1_Y - PIPELINE_H // 2 - 75, W, 25)

# ── PANEL (b): One HopField-Mamba Layer Detail ─────────────────────
LABEL_OFFSET_Y = 30
LAYER_BOX_X = START_X - 20
LAYER_BOX_W = len(boxes_a) * (PIPELINE_W + GAP) - GAP + 40
LAYER_BOX_H = 260
LAYER_BOX_Y = L2_Y - 10
cell("b_bound", "",
     "rounded=1;whiteSpace=wrap;html=1;fillColor=" + NAVY_PALE + ";"
     "strokeColor=" + NAVY_LIGHT + ";strokeWidth=1.5;dashed=0;"
     "fontFamily=Helvetica;fontSize=12;",
     LAYER_BOX_X, LAYER_BOX_Y, LAYER_BOX_W, LAYER_BOX_H)

# Layer title
cell("b_title", "One HopField-Mamba Layer (at token step t)",
     "text;html=1;strokeColor=none;fillColor=none;align=center;"
     "verticalAlign=middle;fontSize=14;fontStyle=1;fontFamily=Helvetica;"
     "fontColor=" + NAVY_MED + ";",
     LAYER_BOX_X, LAYER_BOX_Y - 25, LAYER_BOX_W, 22)

# Mamba SSM block
SSM_X = LAYER_BOX_X + 80
SSM_Y = LAYER_BOX_Y + 40
SSM_W = 150
SSM_H = 70
cell("b_ssm", "Mamba SSM\nĀₜ·hₜ₋₁ + B̄ₜ·xₜ",
     "rounded=1;whiteSpace=wrap;html=1;fillColor=" + TEAL + ";"
     "fontColor=#FFFFFF;fontSize=12;fontStyle=1;fontFamily=Helvetica;"
     "arcSize=14;strokeColor=" + TEAL + ";shadow=1;",
     SSM_X, SSM_Y, SSM_W, SSM_H)

# Gate block
GATE_X = SSM_X + SSM_W + 80
GATE_Y = SSM_Y
GATE_W = 130
GATE_H = 70
cell("b_gate", "Gate σ(·)\n(0, 1)",
     "rounded=1;whiteSpace=wrap;html=1;fillColor=" + ORANGE + ";"
     "fontColor=#FFFFFF;fontSize=12;fontStyle=1;fontFamily=Helvetica;"
     "arcSize=14;strokeColor=" + ORANGE + ";shadow=1;",
     GATE_X, GATE_Y, GATE_W, GATE_H)

# Input arrow x_t
INP_X = SSM_X - 80
edge("b_in", "a0", "b_ssm", "x_t",
     "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
     "jettySize=auto;html=1;strokeWidth=1.5;strokeColor=" + GRAY_LINE + ";"
     "endArrow=blockThin;endSize=10;"
     "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")

# Output arrow h_t
edge("b_out", "b_gate", "a4", "h_t",
     "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
     "jettySize=auto;html=1;strokeWidth=1.5;strokeColor=" + NAVY_LIGHT + ";"
     "endArrow=blockThin;endSize=10;"
     "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")

# SSM → Gate arrow
edge("b_sg", "b_ssm", "b_gate", "",
     "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
     "jettySize=auto;html=1;strokeWidth=1.5;strokeColor=" + GRAY_LINE + ";"
     "endArrow=blockThin;endSize=10;"
     "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")

# g_t control arrow (SSM → Gate bottom)
edge("b_gt", "b_ssm", "b_gate", "gₜ",
     "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
     "jettySize=auto;html=1;strokeWidth=1.5;strokeColor=" + ORANGE_LIGHT + ";"
     "endArrow=blockThin;endSize=10;dashed=1;"
     "exitX=0.5;exitY=1;entryX=0.5;entryY=1;")

# Hopfield Memory block
MEM_X = SSM_X + (GATE_X - SSM_X) // 2 - 40
MEM_Y = SSM_Y + SSM_H + 40
MEM_W = 300
MEM_H = 90
cell("b_mem", "",
     "rounded=1;whiteSpace=wrap;html=1;fillColor=#EBF5FB;"
     "strokeColor=" + NAVY_LIGHT + ";strokeWidth=1.5;dashed=1;"
     "fontFamily=Helvetica;fontSize=12;shadow=1;",
     MEM_X, MEM_Y, MEM_W, MEM_H)
cell("b_mem_title", "Hopfield Memory M  ∈  ℝ^{P×d}",
     "text;html=1;strokeColor=none;fillColor=none;align=center;"
     "verticalAlign=middle;fontSize=13;fontStyle=1;fontFamily=Helvetica;"
     "fontColor=" + NAVY_MED + ";",
     MEM_X, MEM_Y + 4, MEM_W, 22)

# Sub-boxes inside memory: Query, Keys, Values
SUB_W = 70
SUB_H = 30
SUB_GAP = 20
SUB_Y = MEM_Y + 35
sub_total = 3 * SUB_W + 2 * SUB_GAP
sub_start = MEM_X + (MEM_W - sub_total) // 2

sub_labels = ["Query", "Keys", "Values"]
sub_colors = ["#A8D0E6", "#85C1E9", "#5DADE2"]
for i, (lbl, col) in enumerate(zip(sub_labels, sub_colors)):
    sx = sub_start + i * (SUB_W + SUB_GAP)
    cell(f"b_sub{i}", lbl,
         f"rounded=1;whiteSpace=wrap;html=1;fillColor={col};"
         f"fontColor=#1B2A4A;fontSize=11;fontStyle=0;fontFamily=Helvetica;"
         f"arcSize=8;strokeColor={col};",
         sx, SUB_Y, SUB_W, SUB_H)

# Sub-box arrows
for i in range(2):
    edge(f"b_sub_arrow{i}", f"b_sub{i}", f"b_sub{i+1}", "",
         "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
         "jettySize=auto;html=1;strokeWidth=1;strokeColor=" + GRAY_LINE + ";"
         "endArrow=blockThin;endSize=8;"
         f"exitX=1;exitY=0.5;entryX=0;entryY=0.5;")

# Memory read arrow (SSM → Memory)
edge("b_read", "b_ssm", "b_mem", "read",
     "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
     "jettySize=auto;html=1;strokeWidth=1.5;strokeColor=" + NAVY_LIGHT + ";"
     "endArrow=blockThin;endSize=10;"
     "exitX=0.3;exitY=1;entryX=0.3;entryY=0;")

# Memory write arrow (Gate → Memory)
edge("b_write", "b_gate", "b_mem", "write",
     "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
     "jettySize=auto;html=1;strokeWidth=1.5;strokeColor=" + NAVY_LIGHT + ";"
     "endArrow=blockThin;endSize=10;"
     "exitX=0.7;exitY=1;entryX=0.7;entryY=0;")

# Panel (b) label
cell("blabel", "Part (b) — One HopField-Mamba Layer (token step t)",
     "text;html=1;strokeColor=none;fillColor=none;align=center;"
     "verticalAlign=middle;fontSize=15;fontStyle=1;fontFamily=Helvetica;"
     "fontColor=" + GRAY_TEXT + ";",
     0, L2_Y - LAYER_BOX_H // 2 - 55, W, 25)

# ── Write .drawio file ────────────────────────────────────────────
tree = ET.ElementTree(mxGraphModel)
drawio_path = os.path.join(OUT_DIR, "hopfield_mamba_arch.drawio")
tree.write(drawio_path, xml_declaration=True, encoding="UTF-8")
print(f"✓ Wrote {drawio_path}")

# ── Export to PNG using drawio CLI + xvfb ─────────────────────────
png_path = os.path.join(OUT_DIR, "hopfield_mamba_arch.png")
drawio_bin = "/tmp/squashfs-root/drawio"

try:
    from xvfbwrapper import Xvfb
    with Xvfb(width=1920, height=1080) as xvfb:
        env = os.environ.copy()
        env["DISPLAY"] = f":{xvfb.new_display}"
        result = subprocess.run(
            [drawio_bin, "--no-sandbox", "--export", "--format", "png",
             "--output", png_path, "--width", "1600", "--border", "20",
             "--scale", "2",
             drawio_path],
            env=env, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            size = os.path.getsize(png_path) if os.path.exists(png_path) else 0
            print(f"✓ Exported PNG ({size // 1024} KB) → {png_path}")
        else:
            print(f"✗ Export failed:\n{result.stderr[:500]}")
except ModuleNotFoundError:
    print("✗ xvfbwrapper not installed. Cannot run headless drawio.")
except Exception as e:
    print(f"✗ Export error: {e}")
