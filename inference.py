import os
import sys
import json
from openai import OpenAI


def load_local_env() -> None:
    """Lightweight .env loader for local runs without requiring python-dotenv."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_local_env()

# Add the envs module to path so we can import client and models
sys.path.append(os.path.join(os.path.dirname(__file__), "envs", "rag_optimizer_env"))
from client import RagOptimizerEnvClient
from models import RagOptimizerAction

# Load environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

MAX_STEPS = 30
TASK_IDS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an automated Data Engineer managing an AI Knowledge Base.
Your goal is to optimize the messy chunks of text in the database so that a TF-IDF Search Algorithm can find answers easily.
You must resolve contradictions, categorize documents, and delete unnecessary documents.

After each action you will receive a "current_reward" score (0.01 to 0.99) indicating how well the KB currently performs. Use this to guide your strategy.

You have the following actions:
- {"action_type": "read_document", "doc_id": "..."}
- {"action_type": "update_document", "doc_id": "...", "text": "..."}
- {"action_type": "delete_document", "doc_id": "..."}
- {"action_type": "add_metadata", "doc_id": "...", "metadata_key": "...", "metadata_value": "..."}
- {"action_type": "submit"}

You must return ONLY a raw JSON object detailing the action you want to take!"""

def format_action_str(action: RagOptimizerAction) -> str:
    if action.action_type == "read_document":
        return f"read('{action.doc_id}')"
    elif action.action_type == "update_document":
        return f"update('{action.doc_id}')"
    elif action.action_type == "delete_document":
        return f"delete('{action.doc_id}')"
    elif action.action_type == "add_metadata":
        return f"add_metadata('{action.doc_id}','{action.metadata_key}')"
    elif action.action_type == "submit":
        return "submit()"
    return f"{action.action_type}()"

# --- Reflexion (Long Term Memory) ---
LESSONS_FILE = os.path.join(os.path.dirname(__file__), "memory", "lessons_learned.json")

def load_lessons():
    if os.path.exists(LESSONS_FILE):
        try:
            with open(LESSONS_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return []

def save_lesson(lesson_text, task_id):
    os.makedirs(os.path.dirname(LESSONS_FILE), exist_ok=True)
    lessons = load_lessons()
    lessons.append({"task": task_id, "lesson": lesson_text})
    with open(LESSONS_FILE, "w") as f:
        json.dump(lessons, f, indent=2)

def get_system_prompt():
    prompt = SYSTEM_PROMPT
    lessons = load_lessons()
    if lessons:
        prompt += "\n\nPAST LESSONS LEARNED (DO NOT REPEAT MISTAKES):\n"
        for l in lessons[-5:]:  # Show only top 5 recent
            prompt += f"- {l['lesson']}\n"
    return prompt


def _safe_reset(env: RagOptimizerEnvClient, task_id: str):
    """Reset env for a specific task with compatibility fallbacks."""
    try:
        return env.reset(task_id=task_id)
    except TypeError:
        try:
            return env.reset(task=task_id)
        except TypeError:
            return env.reset()


def _clamp_score(value: float) -> float:
    if value < 0.01:
        return 0.01
    if value > 0.99:
        return 0.99
    return value


def run_task_episode(
    env: RagOptimizerEnvClient,
    llm_client: OpenAI,
    task_id: str,
) -> None:
    step_rewards = []
    success = False
    error_msg = "null"
    score = 0.01
    step = 0

    print(f"[START] task={task_id} env=OpenEnv model={MODEL_NAME}")

    # We suppress any other custom prints to respect the STDOUT format strictly
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        try:
            result = _safe_reset(env, task_id)
            observation = result.observation
        except Exception as e:
            error_msg = str(e).replace('\n', ' ')
            print(f"[END] success=false steps=0 score=0.01 rewards=")
            return

    history = [{"role": "system", "content": get_system_prompt()}]

    init_obs = {
        "server_feedback": observation.message,
        "current_reward": observation.reward,
        "current_knowledge_base": observation.current_docs,
    }
    history.append({"role": "user", "content": json.dumps(init_obs, indent=2)})

    for i in range(1, MAX_STEPS + 1):
        step = i
        messages = list(history)

        action_str = "unknown"
        error_msg = "null"

        try:
            completion = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=1000,
            )
            response_text = completion.choices[0].message.content or ""
            action_data = json.loads(response_text)

            # Normalize fields if model returns lists instead of strings
            for field in ("doc_id", "text", "metadata_key", "metadata_value"):
                val = action_data.get(field)
                if isinstance(val, list):
                    if val and isinstance(val[0], str):
                        action_data[field] = " ".join(val)
                    elif val and isinstance(val[0], dict):
                        action_data[field] = json.dumps(val[0])
                    else:
                        action_data[field] = str(val[0]) if val else ""

            action = RagOptimizerAction(**action_data)
            action_str = format_action_str(action)

        except Exception as exc:
            error_msg = str(exc).replace('\n', ' ')
            action = RagOptimizerAction(action_type="submit")
            action_str = format_action_str(action)

        # Suppress normal prints during step
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                result = env.step(action)
                observation = result.observation
                reward = _clamp_score(float(result.reward))
            except Exception as e:
                error_msg = str(e).replace('\n', ' ')
                reward = 0.01
                result = type("obj", (object,), {"done": True})()
                observation = type("obj", (object,), {"message": "error", "current_docs": {}})()

        step_rewards.append(reward)
        done = "true" if result.done else "false"

        print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done} error={error_msg}")

        if result.done:
            success = True if reward > 0.5 else False
            score = _clamp_score(float(reward))
            break

        history.append({"role": "assistant", "content": json.dumps(action.model_dump(), default=str)})
        next_obs = {
            "server_feedback": observation.message,
            "current_reward": observation.reward,
            "current_knowledge_base": observation.current_docs,
        }
        history.append({"role": "user", "content": json.dumps(next_obs, indent=2)})
    else:
        # Reached max steps
        success = False
        score = _clamp_score(float(result.reward))

    rewards_str = ",".join([f"{r:.2f}" for r in step_rewards])
    done_str = "true" if success else "false"
    print(f"[END] success={done_str} steps={step} score={score:.2f} rewards={rewards_str}")

    # Memory Reflexion Trigger
    if not success and score < 0.6:
        # Agent failed, try to reflect
        hist_str = json.dumps([m["content"] for m in history[-6:]])  # get last few actions/obs
        ref_prompt = f"The agent failed task '{task_id}' with final reward {score}. Last context: {hist_str}. Write a 1-sentence tactical lesson stating explicitly what data engineering action the agent should have done instead."
        try:
            resp = llm_client.chat.completions.create(
                model=MODEL_NAME, 
                messages=[{"role": "user", "content": ref_prompt}], 
                max_tokens=60
            )
            lesson = resp.choices[0].message.content.strip()
            save_lesson(lesson, task_id)
            print(f"[MEMORY] Learned lesson: {lesson}")
        except:
            pass

def main():
    # Setup OpenAI Client
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    with RagOptimizerEnvClient(base_url="http://localhost:8000").sync() as env:
        for task_id in TASK_IDS:
            run_task_episode(env=env, llm_client=client, task_id=task_id)

if __name__ == "__main__":
    main()
