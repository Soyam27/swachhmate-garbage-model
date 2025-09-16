from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
from app.services.detection import warmup_model


def create_app() -> FastAPI:
	"""Application factory used by ASGI servers.

	Uvicorn launch examples:
	  uvicorn app.main:app
	  uvicorn app.main:create_app --factory
	"""
	application = FastAPI(title="Garbage Detection API")

	# CORS configuration: adjust allowed origins as needed.
	origins = [
		"*"  # Wide-open by default; replace with specific domains in production.
	]
	application.add_middleware(
		CORSMiddleware,
		allow_origins=origins,
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)

	application.include_router(router, prefix="/api")

	# Warm up the model in the background after startup
	@application.on_event("startup")
	async def _startup_warmup():  # pragma: no cover
		warmup_model()

	return application


# Instantiate a module-level app for simple `uvicorn app.main:app` usage
app = create_app()

if __name__ == "__main__":  # pragma: no cover
	import uvicorn
	uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
