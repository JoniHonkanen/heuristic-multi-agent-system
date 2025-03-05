from prompts.prompts import HEURISTIC_PROMPT
from schemas import AgentState, Code
from .common import cl, llm_code


async def apply_heuristic_logic(state: AgentState) -> Code:
    inputs = state["purpose"]

    # Select the appropriate heuristic prompt
    prompt = HEURISTIC_PROMPT.format(
        user_summary=inputs.user_summary,
        problem_type=inputs.problem_type,
        optimization_focus=inputs.optimization_focus,
        data=state["promptFiles"],
        resource_requirements=inputs.resource_requirements,
    )

    # Interact with the LLM
    structured_llm = llm_code.with_structured_output(Code)
    response = structured_llm.invoke(prompt)

    return response


@cl.step(name="Heuristic Agent")
async def heuristic_agent(state: AgentState) -> AgentState:
    print("*** HEURISTIC AGENT ***")
    current_step = cl.context.current_step

    inputs = state["purpose"]
    current_step.input = (
        f"Applying heuristic optimization based on the following inputs:\n\n"
        f"User Summary: {inputs.user_summary}\n"
        f"Problem Type: {inputs.problem_type}\n"
        f"Optimization Focus: {inputs.optimization_focus}\n"
    )

    try:
        response = await apply_heuristic_logic(state)
    except Exception as e:
        await cl.Message(content=str(e)).send()
        return state

    # Päivitetään output Chainlitin näkymään
    current_step.output = (
        f"Heuristic optimization applied. Suggested solution:\n```python\n{response.python_code}\n```",
        f"Requirements:\n```\n{response.requirements}\n```",
        f"Resources:\n```\n{response.resources}\n```",
    )

    # Tallennetaan heuristinen ratkaisu samaan tapaan kuin generoidut koodit
    state["code"] = response

    with open("generated/generated.py", "w", encoding="utf-8") as f:
        f.write(response.python_code)

    with open("generated/requirements.txt", "w", encoding="utf-8") as f:
        f.write(response.requirements)

    return state
