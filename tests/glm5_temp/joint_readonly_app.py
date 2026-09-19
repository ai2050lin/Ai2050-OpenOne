"""Low-memory local browser acceptance surface during sequential model runs."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.rdc_joint_service import router
app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=['http://127.0.0.1:5173','http://localhost:5173'],allow_methods=['GET'],allow_headers=['*'])
app.include_router(router)
