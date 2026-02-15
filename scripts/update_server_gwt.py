import os

path = r'd:\develop\TransformerLens-main\server\server.py'
if not os.path.exists(path):
    print(f"File {path} not found")
    exit(1)

content = open(path, 'r', encoding='utf-8').read()

insertion = """
@app.get("/nfb/gwt/status")
async def get_gwt_status():
    return global_workspace_controller.locus_of_attention

@app.post("/nfb/gwt/update")
async def update_gwt(layer_idx: int, x: float, y: float, z: float, intensity: float):
    return global_workspace_controller.update_locus(layer_idx, [x, y, z], intensity)
"""

if '@app.get("/health")' in content:
    new_content = content.replace('@app.get("/health")', insertion + '\n@app.get("/health")')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully updated server/server.py with GWT endpoints.")
else:
    print("Could not find health check endpoint in server/server.py")
