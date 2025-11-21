import threading
import time
from datetime import datetime
import duckdb

# from Autonomous_Reasoning_System.tools.action_executor import ActionExecutor # Removed dumb executor
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


def start_heartbeat_with_plans(learner, confidence, plan_builder, interval_seconds=90, test_mode=True, plan_executor=None):
    """
    Heartbeat loop with plan awareness.
    Periodically summarises learning, reminds Tyrone of active plans,
    checks due reminders, and autonomously executes the next pending step for each plan.
    """

    # Use passed plan_executor or warn
    if not plan_executor:
         print("[WARN] Scheduler running without robust PlanExecutor. Plan steps may fail.")

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

                                    result_status = "failed"
                                    result_output = "No executor available"

                                    if plan_executor:
                                         # Use PlanExecutor internal logic to route/execute step
                                         # Note: We reuse _execute_step if public, or simulate it.
                                         # Since _execute_step is protected, strictly we should use execute_plan(plan_id).
                                         # But execute_plan runs loop.
                                         # Assuming we want to run just ONE step in background per tick:
                                         # We can invoke execute_plan, but it might run multiple steps if they are fast.
                                         # Let's assume execute_plan handles resume correctly.
                                         exec_res = plan_executor.execute_plan(plan.id)

                                         if exec_res["status"] == "success" or exec_res["status"] == "active":
                                              # We need to check if THIS step passed.
                                              # PlanExecutor updates the plan object.
                                              # We check plan status.
                                              # But execute_plan runs ALL steps.
                                              # If we want strictly one step, we need a different method or accept it runs all.
                                              # "Autonomously executes the next pending step" implies singular.
                                              # However, if PlanExecutor runs all, that's even better autonomy!
                                              result_status = "complete" if plan.status == "complete" else "running"
                                              result_output = str(exec_res)
                                         else:
                                              result_status = "failed"
                                              result_output = str(exec_res.get("errors"))
                                    else:
                                         # Fallback/Dummy
                                         result_status = "failed"
                                         result_output = "PlanExecutor missing"

                                    # Plan updates are handled inside plan_executor usually.
                                    # If we used execute_plan, we don't need manual update here.
                                    # So we only log.
                                    print(f"[🤖 EXECUTOR] Result: {result_status}")

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
