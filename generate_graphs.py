# ============================================================
# FraudGuard — Diagram Generator (Windows Local / Anaconda)
# ============================================================
# BEFORE RUNNING:
#   1. Install Graphviz for Windows from https://graphviz.org/download/
#      (choose the .exe installer, e.g. graphviz-11.x.x-win64.exe)
#   2. During install, tick "Add Graphviz to PATH" — or add manually:
#      Control Panel > System > Environment Variables > Path
#      Add: C:\Program Files\Graphviz\bin
#   3. Then install the Python binding (if not already):
#      pip install graphviz
#   4. Restart your terminal/Anaconda prompt after installing
# ============================================================

import os
import sys
from graphviz import Digraph

# Quick sanity check — will raise a clear error if Graphviz isn't on PATH
import shutil
if shutil.which('dot') is None:
    sys.exit(
        "\nERROR: Graphviz 'dot' executable not found on PATH.\n"
        "Please install Graphviz from https://graphviz.org/download/\n"
        "and make sure its 'bin' folder is added to your system PATH.\n"
        "After installing, restart your terminal and try again.\n"
    )

os.makedirs('graphs', exist_ok=True)

# ============================================================
# DESIGN SYSTEM
# ============================================================
FONT        = 'Helvetica'
LABEL_COLOR = '#1E293B'
EDGE_COLOR  = '#64748B'

C = {
    'raw':      '#EEF2FF',
    'struct':   '#FFF7ED',
    'freetext': '#F0FDF4',
    'process':  '#EFF6FF',
    'ml':       '#EFF6FF',
    'backend':  '#FFF7ED',
    'frontend': '#F5F3FF',
    'infra':    '#F0FDF4',
    'input':    '#F8FAFC',
    'model':    '#EFF6FF',
    'ext':      '#F5F3FF',
    'output':   '#FFF1F2',
    'embed':    '#FFFBEB',
    'encoder':  '#EFF6FF',
    'head':     '#F5F3FF',
}

N = {
    'default': '#FFFFFF',
    'process': '#DBEAFE',
    'output':  '#FEE2E2',
    'fraud':   '#FCA5A5',
    'legit':   '#86EFAC',
}

BORDER = {
    'struct':   '#D97706',
    'freetext': '#16A34A',
    'process':  '#2563EB',
    'ml':       '#2563EB',
    'backend':  '#D97706',
    'frontend': '#7C3AED',
    'infra':    '#16A34A',
    'input':    '#94A3B8',
    'model':    '#2563EB',
    'ext':      '#7C3AED',
    'output':   '#DC2626',
    'embed':    '#B45309',
    'encoder':  '#2563EB',
    'head':     '#7C3AED',
}

def base_node_attrs():
    return dict(
        shape='box', style='filled,rounded',
        fillcolor=N['default'],
        fontname=FONT, fontcolor=LABEL_COLOR,
        fontsize='9', color='#CBD5E1',
        width='1.1', height='0.35', margin='0.12,0.07',
    )

def base_edge_attrs():
    return dict(
        color=EDGE_COLOR, arrowsize='0.7', penwidth='1.2',
        fontname=FONT, fontsize='8', fontcolor='#475569',
    )

def cluster_attrs(key, label):
    return dict(
        label=f'  {label}  ',
        style='filled,rounded',
        fillcolor=C[key],
        color=BORDER.get(key, '#CBD5E1'),
        fontname=FONT, fontsize='9.5',
        fontcolor=LABEL_COLOR, penwidth='1.3', margin='10',
    )

# ============================================================
# FIGURE 1 — DATA PREPROCESSING PIPELINE
# ============================================================
def make_pipeline():
    dot = Digraph('Preprocessing', format='png')
    dot.attr(
        rankdir='TB', splines='spline',
        nodesep='0.5', ranksep='0.7',
        bgcolor='white', fontname=FONT,
        pad='0.3', size='5,7!', dpi='180',
    )
    dot.attr('node', **base_node_attrs())
    dot.attr('edge', **base_edge_attrs())

    dot.node('RAW', 'Raw CSV Row\n(18 columns)',
             fillcolor='#F1F5F9', color='#94A3B8',
             style='filled,rounded', fontsize='9')

    with dot.subgraph(name='cluster_struct') as c:
        c.attr(**cluster_attrs('struct', 'Structured Fields'))
        for nid, lbl in [('S1', 'Has Logo'), ('S2', 'Employment Type'),
                          ('S3', 'Salary Range'), ('S4', 'Location')]:
            c.node(nid, lbl)

    with dot.subgraph(name='cluster_free') as c:
        c.attr(**cluster_attrs('freetext', 'Free-Text Fields'))
        for nid, lbl in [('F1', 'Benefits'), ('F2', 'Requirements'),
                          ('F3', 'Description'), ('F4', 'Company Profile'),
                          ('F5', 'Job Title')]:
            c.node(nid, lbl)

    for nid, lbl in [('JOIN', 'Join with [SEP]'),
                      ('TOKEN', 'BPE Tokenizer'),
                      ('OUT', 'input_ids + attention_mask')]:
        dot.node(nid, lbl, fillcolor=N['process'],
                 color='#93C5FD', fontcolor=LABEL_COLOR)

    dot.edge('RAW', 'S2')
    dot.edge('RAW', 'F3')
    dot.edge('S2', 'JOIN', xlabel='structured')
    dot.edge('F3', 'JOIN', xlabel='free-text')
    dot.edge('JOIN', 'TOKEN')
    dot.edge('TOKEN', 'OUT')

    out = dot.render('graphs/01_pipeline', cleanup=True)
    print(f'[1/4] Pipeline saved → {out}')
    return out

# ============================================================
# FIGURE 2 — TECHNOLOGY STACK
# ============================================================
def make_tech_stack():
    dot = Digraph('TechStack', format='png')
    dot.attr(
        rankdir='LR', splines='spline',
        nodesep='0.5', ranksep='0.9',
        bgcolor='white', fontname=FONT,
        pad='0.3', size='8,4.5!', dpi='180',
    )
    dot.attr('node', **base_node_attrs())
    dot.attr('edge', **base_edge_attrs())

    with dot.subgraph(name='cluster_ml') as c:
        c.attr(**cluster_attrs('ml', 'Machine Learning'))
        for nid, lbl in [('M1', 'PyTorch 2.2'), ('M2', 'Transformers 4.44'),
                          ('M3', 'Optuna HPO'), ('M4', 'scikit-learn')]:
            c.node(nid, lbl)

    with dot.subgraph(name='cluster_backend') as c:
        c.attr(**cluster_attrs('backend', 'Backend'))
        for nid, lbl in [('B1', 'Flask 3.x'), ('B2', 'LangChain'),
                          ('B3', 'OpenRouter LLM'), ('B4', 'FastAPI\nModel API')]:
            c.node(nid, lbl)

    with dot.subgraph(name='cluster_frontend') as c:
        c.attr(**cluster_attrs('frontend', 'Frontend & Extension'))
        for nid, lbl in [('F1', 'Jinja2 Templates'),
                          ('F2', 'Chrome MV3 Extension'),
                          ('F3', 'Google Gemini API')]:
            c.node(nid, lbl)

    with dot.subgraph(name='cluster_infra') as c:
        c.attr(**cluster_attrs('infra', 'Infrastructure'))
        for nid, lbl in [('I1', 'HuggingFace Hub\nModel Weights'),
                          ('I2', 'HuggingFace Spaces\nDocker API'),
                          ('I3', 'DuckDuckGo Search')]:
            c.node(nid, lbl)

    dot.edge('B1', 'B2');  dot.edge('B2', 'B3')
    dot.edge('F2', 'F3')
    dot.edge('M2', 'B4');  dot.edge('B1', 'F1')
    dot.edge('B4', 'I2');  dot.edge('M2', 'I1')

    out = dot.render('graphs/02_tech_stack', cleanup=True)
    print(f'[2/4] Tech Stack saved → {out}')
    return out

# ============================================================
# FIGURE 3 — SYSTEM ARCHITECTURE
# ============================================================
def make_architecture():
    dot = Digraph('FraudGuard_Architecture', format='png')
    dot.attr(
        rankdir='LR', splines='spline',
        nodesep='0.45', ranksep='0.85',
        bgcolor='white', fontname=FONT,
        pad='0.3', size='9,5!', dpi='180',
    )
    dot.attr('node', **base_node_attrs())
    dot.attr('edge', **base_edge_attrs())

    with dot.subgraph(name='cluster_input') as c:
        c.attr(**cluster_attrs('input', 'Input Layer'))
        for nid, lbl in [('A1', 'Raw Text'), ('A2', 'File Upload'),
                          ('A3', 'LinkedIn URL')]:
            c.node(nid, lbl)

    with dot.subgraph(name='cluster_web') as c:
        c.attr(**cluster_attrs('backend', 'Flask Web App'))
        for nid, lbl in [('B1', 'Extractor'), ('B2', 'Research'),
                          ('B3', 'Verification'), ('B4', 'LLM Summary'),
                          ('B5', 'Final Report')]:
            c.node(nid, lbl)

    with dot.subgraph(name='cluster_model') as c:
        c.attr(**cluster_attrs('model', 'RoBERTa Model'))
        for nid, lbl in [('C1', 'Tokenizer'), ('C2', 'RoBERTa'),
                          ('C3', 'Classifier'), ('C4', 'Threshold')]:
            c.node(nid, lbl)
        c.node('C5', 'Fraud', fillcolor=N['fraud'],
               color='#F87171', fontcolor=LABEL_COLOR)
        c.node('C6', 'Legit', fillcolor=N['legit'],
               color='#4ADE80', fontcolor=LABEL_COLOR)

    with dot.subgraph(name='cluster_ext') as c:
        c.attr(**cluster_attrs('ext', 'Chrome Extension'))
        for nid, lbl in [('D1', 'DOM Scraper'), ('D2', 'Gemini API'),
                          ('D3', 'Overlay UI')]:
            c.node(nid, lbl)

    with dot.subgraph(name='cluster_out') as c:
        c.attr(**cluster_attrs('output', 'Output'))
        c.node('E1', 'Web Report',
               fillcolor=N['output'], color='#FCA5A5', fontcolor=LABEL_COLOR)
        c.node('E2', 'Browser Overlay',
               fillcolor=N['output'], color='#FCA5A5', fontcolor=LABEL_COLOR)

    dot.edge('A2', 'B1')
    dot.edge('B1', 'B2'); dot.edge('B2', 'B3')
    dot.edge('B3', 'B4'); dot.edge('B4', 'B5')
    dot.edge('B1', 'C1', style='dashed', xlabel='text')
    dot.edge('C1', 'C2'); dot.edge('C2', 'C3'); dot.edge('C3', 'C4')
    dot.edge('C4', 'C5', xlabel='Fraud')
    dot.edge('C4', 'C6', xlabel='Legit')
    dot.edge('C5', 'B5', style='dashed')
    dot.edge('C6', 'B5', style='dashed')
    dot.edge('B5', 'E1')
    dot.edge('A3', 'D1'); dot.edge('D1', 'D2')
    dot.edge('D2', 'D3'); dot.edge('D3', 'E2')

    out = dot.render('graphs/03_architecture', cleanup=True)
    print(f'[3/4] Architecture saved → {out}')
    return out

# ============================================================
# FIGURE 4 — ROBERTA MODEL ARCHITECTURE
# ============================================================
def make_roberta():
    dot = Digraph('RoBERTa_Architecture', format='png')
    dot.attr(
        rankdir='TB', splines='spline',
        nodesep='0.45', ranksep='0.65',
        bgcolor='white', fontname=FONT,
        pad='0.3', size='4.5,8!', dpi='180',
    )
    dot.attr('node', **base_node_attrs())
    dot.attr('edge', **base_edge_attrs())

    for nid, lbl, fc, bc in [
        ('INPUT',  'Token Sequence\n[CLS] ... [SEP]', '#F1F5F9', '#94A3B8'),
        ('CLS',    '[CLS] Representation',            N['process'], '#93C5FD'),
        ('THRESH', 'Threshold  P >= 0.87',            '#FEF9C3',   '#FCD34D'),
        ('OUT1',   'FRAUDULENT',                       N['fraud'],  '#F87171'),
        ('OUT2',   'LEGITIMATE',                       N['legit'],  '#4ADE80'),
    ]:
        dot.node(nid, lbl, fillcolor=fc, color=bc, fontcolor=LABEL_COLOR)

    with dot.subgraph(name='cluster_embed') as c:
        c.attr(**cluster_attrs('embed', 'Embedding Layer'))
        for nid, lbl in [('E1', 'Token Embeddings'),
                          ('E2', 'Position Embeddings'),
                          ('E3', 'LayerNorm + Dropout')]:
            c.node(nid, lbl)

    with dot.subgraph(name='cluster_enc') as c:
        c.attr(**cluster_attrs('encoder', 'RoBERTa Encoder  (x12 Layers)'))
        for nid, lbl in [('L1', 'Multi-Head Attention'),
                          ('L2', 'Feed Forward'),
                          ('L3', '... (x12 layers)')]:
            c.node(nid, lbl)

    with dot.subgraph(name='cluster_head') as c:
        c.attr(**cluster_attrs('head', 'Classification Head'))
        for nid, lbl in [('D',   'Dropout'),
                          ('LIN', 'Linear  768 -> 2'),
                          ('SM',  'Softmax')]:
            c.node(nid, lbl)

    dot.edge('INPUT', 'E1'); dot.edge('E1', 'E2'); dot.edge('E2', 'E3')
    dot.edge('E3',    'L1'); dot.edge('L1', 'L2'); dot.edge('L2', 'L3')
    dot.edge('L3',    'CLS')
    dot.edge('CLS',   'D');  dot.edge('D',   'LIN'); dot.edge('LIN', 'SM')
    dot.edge('SM',    'THRESH')
    dot.edge('THRESH', 'OUT1', xlabel='Fraud')
    dot.edge('THRESH', 'OUT2', xlabel='Legit')

    out = dot.render('graphs/04_roberta', cleanup=True)
    print(f'[4/4] RoBERTa saved → {out}')
    return out

# ============================================================
# RUN ALL
# ============================================================
if __name__ == '__main__':
    print('Generating diagrams...\n')
    make_pipeline()
    make_tech_stack()
    make_architecture()
    make_roberta()
    print('\nAll 4 diagrams saved to the graphs/ folder.')
