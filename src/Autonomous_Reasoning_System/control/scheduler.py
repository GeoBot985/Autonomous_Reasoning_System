import threading
import time
import logging
from datetime import datetime
from Autonomous_Reasoning_System.infrastructure.observability import Metrics
# from Autonomous_Reasoning_System.tools.action_executor import ActionExecutor # Removed dumb executor
from Autonomous_Reasoning_System.control.attention_manager import attention  # 🧭 added

logger = logging.getLogger(__name__)

lock = threading.Lock()  # global lock shared by the thread


def check_due_reminders(memory_storage, lookahead_minutes=1):
    """
    Scan stored memories for any 'task' entries due within ±lookahead_minutes,
    print reminders once, and mark them as 'triggered' to avoid repeats.
    """
    try:
        df = memory_storage.get_all_memories()
        if df.empty or "scheduled_for" not in df.columns:
            return

        now = datetime.utcnow()

        # Select untriggered reminders due now or very soon
        due = df[
            (df["memory_type"] == "task")
            & (df["scheduled_for"].notna())
            & (df["status"].isna() | (df["status"] != "triggered"))
            & ((df["scheduled_for"] - now).dt.total_seconds().abs() < lookahead_minutes * 60)
        ]

        if due.empty:
            return

        for _, row in due.iterrows():
            logger.info(f"⏰ Reminder: {row['text']} (scheduled {row['scheduled_for']})")

            # Mark reminder as triggered so it fires only once
            try:
                with memory_storage._write_lock:
                    memory_storage.con.execute(
                        "UPDATE memory SET status = 'triggered' WHERE id = ?",
                        (row["id"],)
                    )
                logger.info(f"✅ Marked reminder '{row['text'][:40]}...' as triggered.")
            except Exception as e:
                logger.warning(f"[⚠️ ReminderUpdate] Failed to mark triggered: {e}")

    except Exception as e:
        logger.error(f"[⚠️ ReminderCheck] {e}")


def start_heartbeat_with_plans(learner, confidence, plan_builder, interval_seconds=90, test_mode=True, plan_executor=None):
    """
    Heartbeat loop with plan awareness.
    Periodically summarises learning, reminds Tyrone of active plans,
    checks due reminders, and autonomously executes the next pending step for each plan.
    """

    # Use passed plan_executor or warn
    if not plan_executor:
         logger.warning("[WARN] Scheduler running without robust PlanExecutor. Plan steps may fail.")

    def loop():
        time.sleep(3)  # let systems initialise first
        counter = 0
        while True:
            try:
                start_tick = time.time()
                # 🧭 Attention Check — skip background work if user is active or recently interacted
                if attention.should_pause_autonomous():
                    # optional: only print occasionally to avoid clutter
                    # logger.info("[🧭 ATTENTION] User or recent activity detected — pausing background tasks.")
                    time.sleep(5)
                    continue

                with lock:  # prevent overlap
                    Metrics().increment("scheduler_heartbeat")
                    # --- learning summary ---
                    summary = learner.summarise_recent(window_minutes=2)
                    ts = datetime.now().strftime("%H:%M:%S")
                    logger.info(f"[🕒 HEARTBEAT] {ts} → {summary['summary']}")
                    if hasattr(confidence, "decay_all"):
                        confidence.decay_all()

                    # --- reminder check ---
                    check_due_reminders(learner.memory_storage if hasattr(learner, "memory_storage") else learner.memory)

                    # --- every few pulses, check active plans ---
                    counter += 1
                    if counter % 3 == 0:  # e.g. every 3 heartbeats
                        # Check if plan_builder has get_active_plans, otherwise use memory
                        if hasattr(plan_builder, 'get_active_plans'):
                            active = plan_builder.get_active_plans()
                        else:
                            active = plan_builder.memory.get_active_plans()

                        if active:
                            logger.info(f"[📋 ACTIVE PLANS] {len(active)} ongoing:")
                            for plan in active:
                                # Determine progress (simple approximation based on status string)
                                progress_desc = f"{len(plan.steps)} steps"

                                logger.info(f"   • {plan.goal}: Status {plan.status} ({progress_desc})")

                                # 🧠 store reflection reminder
                                # Use memory directly if available (MemoryStorage has 'remember')
                                mem = getattr(plan_builder, "memory", None)
                                if mem and hasattr(mem, "remember"):
                                    mem.remember(
                                        text=f"Reminder: Continue plan '{plan.goal}'. Status: {plan.status}.",
                                        memory_type="plan_reminder",
                                        importance=0.3,
                                        source="Scheduler"
                                    )

                                # 🤖 attempt next step automatically
                                if plan_executor:
                                     logger.info(f"[🤖 EXECUTOR] Attempting execution for '{plan.goal}'")
                                     # Use PlanExecutor's new execute_next_step method
                                     exec_res = plan_executor.execute_next_step(plan.id)

                                     status = exec_res.get("status")
                                     result_output = ""

                                     if status == "complete":
                                          result_output = "Plan finished!"
                                     elif status == "running":
                                          result_output = f"Step completed: {exec_res.get('step_completed')}"
                                     elif status == "suspended":
                                          result_output = f"Suspended: {exec_res.get('errors')}"
                                     else:
                                          result_output = str(exec_res.get("errors"))

                                     logger.info(f"[🤖 EXECUTOR] Result: {status} - {result_output}")
                                else:
                                     # Fallback: Just log if no executor
                                     logger.warning(f"[🤖 EXECUTOR] Skipping execution for '{plan.goal}' (No executor available)")

                        else:
                            logger.info("[📋 ACTIVE PLANS] None currently active.")

                # Record timing
                Metrics().record_time("scheduler_tick_duration", time.time() - start_tick)

            except Exception as e:
                logger.error(f"[⚠️ HEARTBEAT ERROR] {e}")
                Metrics().increment("scheduler_errors")

            time.sleep(interval_seconds)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    mode = "TEST" if test_mode else "NORMAL"
    logger.info(f"[⏰ HEARTBEAT+PLANS] Started ({mode} mode, interval={interval_seconds}s).")
    return t
