import threading
import time
import logging
from datetime import datetime
import duckdb

# from Autonomous_Reasoning_System.tools.action_executor import ActionExecutor # Removed dumb executor
from Autonomous_Reasoning_System.control.attention_manager import attention  # 🧭 added

logger = logging.getLogger(__name__)

lock = threading.Lock()  # global lock shared by the thread


def check_due_reminders(memory, lookahead_minutes=1):
    """
    Scan stored memories for any 'task' entries due within ±lookahead_minutes,
    print reminders once, and mark them as 'triggered' to avoid repeats.
    """
    try:
        df = memory.get_all_memories()
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
                duckdb.sql(f"""
                    CREATE OR REPLACE TABLE memory_temp AS
                    SELECT * FROM read_parquet('{memory.db_path}');
                """)
                duckdb.sql(f"""
                    UPDATE memory_temp
                    SET status = 'triggered'
                    WHERE id = '{row['id']}';
                """)
                duckdb.sql(
                    f"COPY memory_temp TO '{memory.db_path}' (FORMAT PARQUET, OVERWRITE TRUE);"
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
                # 🧭 Attention Check — skip background work if user is active or recently interacted
                if attention.should_pause_autonomous():
                    # optional: only print occasionally to avoid clutter
                    # logger.info("[🧭 ATTENTION] User or recent activity detected — pausing background tasks.")
                    time.sleep(5)
                    continue

                with lock:  # prevent overlap
                    # --- learning summary ---
                    summary = learner.summarise_recent(window_minutes=2)
                    ts = datetime.now().strftime("%H:%M:%S")
                    logger.info(f"[🕒 HEARTBEAT] {ts} → {summary['summary']}")
                    if hasattr(confidence, "decay_all"):
                        confidence.decay_all()

                    # --- reminder check ---
                    check_due_reminders(learner.memory)

                    # --- every few pulses, check active plans ---
                    counter += 1
                    if counter % 3 == 0:  # e.g. every 3 heartbeats
                        active = plan_builder.get_active_plans()
                        if active:
                            logger.info(f"[📋 ACTIVE PLANS] {len(active)} ongoing:")
                            for plan in active:
                                prog = plan.progress_summary()
                                logger.info(f"   • {plan.title}: {prog['completed_steps']}/{prog['total_steps']} steps complete.")

                                # 🧠 store reflection reminder
                                plan_builder.memory.add_memory(
                                    text=f"Reminder: Continue plan '{plan.title}'. Current step: {prog['current_step']}.",
                                    memory_type="plan_reminder",
                                    importance=0.3,
                                    source="Scheduler"
                                )

                                # 🤖 attempt next step automatically
                                next_step = plan.next_step()
                                if next_step and next_step.status == "pending":
                                    logger.info(f"[🤖 EXECUTOR] Running next step for '{plan.title}': {next_step.description}")

                                    result_status = "failed"
                                    result_output = "No executor available"

                                    if plan_executor:
                                         # Use PlanExecutor's new execute_next_step method
                                         exec_res = plan_executor.execute_next_step(plan.id)

                                         status = exec_res.get("status")
                                         if status == "complete":
                                              result_status = "complete"
                                              result_output = "Plan finished!"
                                         elif status == "running":
                                              result_status = "running"
                                              result_output = f"Step completed: {exec_res.get('step_completed')}"
                                         elif status == "suspended":
                                              result_status = "suspended"
                                              result_output = f"Suspended: {exec_res.get('errors')}"
                                         else:
                                              result_status = "failed"
                                              result_output = str(exec_res.get("errors"))
                                    else:
                                         # Fallback/Dummy
                                         result_status = "failed"
                                         result_output = "PlanExecutor missing"

                                    # Plan updates are handled inside plan_executor usually.
                                    logger.info(f"[🤖 EXECUTOR] Result: {result_status}")

                        else:
                            logger.info("[📋 ACTIVE PLANS] None currently active.")

            except Exception as e:
                logger.error(f"[⚠️ HEARTBEAT ERROR] {e}")

            time.sleep(interval_seconds)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    mode = "TEST" if test_mode else "NORMAL"
    logger.info(f"[⏰ HEARTBEAT+PLANS] Started ({mode} mode, interval={interval_seconds}s).")
    return t
