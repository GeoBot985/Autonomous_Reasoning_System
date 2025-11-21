from Autonomous_Reasoning_System.infrastructure.logging_utils import setup_logging
from Autonomous_Reasoning_System.infrastructure import startup_validator
from Autonomous_Reasoning_System import init_runtime
from Autonomous_Reasoning_System.infrastructure import config
from Autonomous_Reasoning_System.infrastructure.observability import HealthServer, Metrics
from pathlib import Path
import uvicorn


def main():
    setup_logging()

    # Log Startup
    Metrics().increment("system_startup")

    # Startup Protection Layer
    data_dir = Path(config.MEMORY_DB_PATH).parent
    if not data_dir.exists() or not any(data_dir.iterdir()):
         print("[Main] First launch detected or data directory empty. Initializing...")
         init_runtime.bootstrap_runtime()

    startup_validator.validate_startup()

    # Start the API server (blocks)
    from Autonomous_Reasoning_System.infrastructure.api import app
    print("Tyrone API worker started – http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
