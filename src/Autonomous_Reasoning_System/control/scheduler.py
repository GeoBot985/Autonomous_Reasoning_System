import threading
import time
from datetime import datetime
import duckdb

from Autonomous_Reasoning_System.tools.action_executor import ActionExecutor
from Autonomous_Reasoning_System.control.attention_manager import attention  # 🧭 added


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
            print(f"⏰ Reminder: {row['text']} (scheduled {row['scheduled_for']})")

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
                print(f"✅ Marked reminder '{row['text'][:40]}...' as triggered.")
            except Exception as e:
                print(f"[⚠️ ReminderUpdate] Failed to mark triggered: {e}")

    except Exception as e:
        print(f"[⚠️ ReminderCheck] {e}")


def start_heartbeat_with_plans(learner, confidence, plan_builder, interval_seconds=90, test_mode=True):
    """
    Heartbeat loop with plan awareness.
    Periodically summarises learning, reminds Tyrone of active plans,
    checks due reminders, and autonomously executes the next pending step for each plan.
    """
    executor = ActionExecutor()

    def loop():
        time.sleep(3)  # let systems initialise first
        counter = 0
        while True:
            try:
                # 🧭 Attention Check — skip background work if user is active or recently interacted
                if attention.should_pause_autonomous():
                    # optional: only print occasionally to avoid clutter
                    # print("[🧭 ATTENTION] User or recent activity detected — pausing background tasks.")
                    time.sleep(5)
                    continue

                with lock:  # prevent overlap
                    # --- learning summary ---
                    summary = learner.summarise_recent(window_minutes=2)
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[🕒 HEARTBEAT] {ts} → {summary['summary']}")
                    if hasattr(confidence, "decay_all"):
                        confidence.decay_all()

                    # --- reminder check ---
                    check_due_reminders(learner.memory)

                    # --- every few pulses, check active plans ---
                    counter += 1
                    if counter % 3 == 0:  # e.g. every 3 heartbeats
                        active = plan_builder.get_active_plans()
                        if active:
                            print(f"[📋 ACTIVE PLANS] {len(active)} ongoing:")
                            for plan in active:
                                prog = plan.progress_summary()
                                print(f"   • {plan.title}: {prog['completed_steps']}/{prog['total_steps']} steps complete.")

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
                                    print(f"[🤖 EXECUTOR] Running next step for '{plan.title}': {next_step.description}")
                                    result = executor.execute_step(next_step.description, plan.workspace)
                                    status = "complete" if result["success"] else "failed"
                                    plan.mark_step(next_step.id, status, result["result"])
                                    plan_builder.update_step(plan.id, next_step.id, status, result["result"])
                        else:
                            print("[📋 ACTIVE PLANS] None currently active.")

            except Exception as e:
                print(f"[⚠️ HEARTBEAT ERROR] {e}")

            time.sleep(interval_seconds)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    mode = "TEST" if test_mode else "NORMAL"
    print(f"[⏰ HEARTBEAT+PLANS] Started ({mode} mode, interval={interval_seconds}s).")
    return t
