"""Task-owned read-only server, separate from all previous research routes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.rdc_construction_service import router

app = FastAPI(title='RDC contextual query construction and native parameters')
app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:5173', 'http://localhost:5173'],
                   allow_methods=['GET'], allow_headers=['*'])
app.include_router(router)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=5004)
