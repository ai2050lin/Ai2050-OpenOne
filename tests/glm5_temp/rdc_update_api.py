"""Isolated read-only atlas server; leaves the user's existing port5001 app running."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.rdc_update_service import router
app=FastAPI(title='RDC relation-update read-only evidence')
app.add_middleware(CORSMiddleware,allow_origins=['http://127.0.0.1:5173','http://localhost:5173'],allow_methods=['GET'],allow_headers=['*'])
app.include_router(router)

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=5002)
