
import threading

# Global lock for memory write operations to ensure concurrency safety across modules
memory_write_lock = threading.Lock()
