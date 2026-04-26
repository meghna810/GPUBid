"""Render functions returning HTML strings or Plotly/Matplotlib figures.

Designed to be shell-agnostic: the notebook calls `display(HTML(render_market(m)))`,
a future Gradio app calls `gr.HTML(render_market(m))`, a future Next.js app
passes the string into `dangerouslySetInnerHTML`. Same render code, three shells.
"""
