from uagents import Agent, Context, Model
from models import SharedAgentState  # Ensure this matches the frontend model exactly

# If you don't have a mailbox, both must be on the same machine/network
bob = Agent(
    name="Stress_Analyzer",
    port=8002,
    endpoint=["http://127.0.0.1:8002/submit"],
)


def analyze_stress_data(state: SharedAgentState) -> SharedAgentState:
    # Use the decimal 'brightness' sent from frontend
    score_pct = int(state.brightness * 100)

    if score_pct > 70:
        state.result = f"🚨 HIGH CORTISOL ({score_pct}%): Clinical markers indicate acute stress. I recommend the box breathing exercise immediately."
    elif score_pct > 40:
        state.result = f"⚠️ MODERATE CORTISOL ({score_pct}%): You are trending above baseline. Consider a short walk or a screen break."
    else:
        state.result = f"✅ LOW CORTISOL ({score_pct}%): Physiological parameters are optimal. Maintain your current routine."

    return state


@bob.on_message(model=SharedAgentState)
async def handle_message(ctx: Context, sender: str, msg: SharedAgentState):
    ctx.logger.info(f"Received request from {sender}")

    # Process the data
    updated_state = analyze_stress_data(msg)

    # Send the updated state back to the sender (the frontend query)
    await ctx.send(sender, updated_state)


if __name__ == "__main__":
    bob.run()