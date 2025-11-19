from datetime import datetime, timedelta
from Autonomous_Reasoning_System.memory.singletons import get_memory_storage
from Autonomous_Reasoning_System.control.scheduler import check_due_reminders
import time

print("=== 🧠 Reminder System Test ===")

mem = get_memory_storage()

# Schedule a reminder 30 seconds in the future
reminder_time = datetime.utcnow() + timedelta(seconds=30)
text = f"Meeting with Cornelia at {reminder_time.strftime('%H:%M:%S')} UTC"

mem.add_memory(
    text=text,
    memory_type="task",
    importance=0.6,
    source="TestSuite",
    scheduled_for=reminder_time.isoformat(),
)
print(f"✅ Reminder stored for {reminder_time}")

# Wait and check reminders every 10 seconds
for i in range(1, 5):
    print(f"\n[⏳ Loop {i}] Checking reminders...")
    check_due_reminders(mem, lookahead_minutes=1)
    time.sleep(10)

print("\n=== ✅ Test complete ===")
