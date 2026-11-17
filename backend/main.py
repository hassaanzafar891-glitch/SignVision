
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import capture, recognition, speech, stream, contact, browser_camera, monitoring
from core.config import settings
from monitoring import MonitoringMiddleware, get_monitor

app = FastAPI(title="PSL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MonitoringMiddleware, monitor=get_monitor())

app.include_router(capture.router,     prefix="/api", tags=["capture"])
app.include_router(recognition.router, prefix="/api", tags=["recognition"])
app.include_router(speech.router,      prefix="/api", tags=["speech"])
app.include_router(stream.router,      prefix="/api", tags=["stream"])
app.include_router(contact.router,     prefix="/api", tags=["contact"])
app.include_router(browser_camera.router, prefix="/api", tags=["browser-camera"])
app.include_router(monitoring.router, prefix="/api")


@app.get("/")
def root():
    return {"status": "PSL API running"}
