import ast
import math
import operator

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# TOOL 1: CALCULATOR
# =====================================================

@tool
def calculator(expression: str) -> str:
    """Calculate arithmetic expressions such as 25*4, 100/5, or 2**3."""

    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent is too large")

            result = operations[type(node.op)](left, right)

            if isinstance(result, (int, float)) and abs(result) > 1e100:
                raise ValueError("Result is too large")

            return result

        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value

        raise ValueError("Unsupported mathematical expression")

    try:
        tree = ast.parse(expression, mode="eval")
        answer = evaluate(tree)

        if not math.isfinite(answer):
            return "Error: Result is not finite."

        return f"Calculation result: {answer}"

    except Exception as error:
        return f"Calculation error: {error}"


# =====================================================
# TOOL 2: FAQ RETRIEVAL
# =====================================================

faq_data = [
    (
        "computer running slowly",
        "Close unnecessary applications, check available storage, "
        "disable unnecessary startup programs, and scan for malware."
    ),
    (
        "computer not turning on",
        "Check the wall socket, power cable, charger, battery, "
        "and power supply. If the problem persists, seek repair."
    ),
    (
        "laptop overheating",
        "Keep the ventilation openings clear, place the laptop "
        "on a hard surface, and check whether its fans are working. "
        "Shut it down if it becomes dangerously hot."
    ),
    (
        "wifi not connecting",
        "Check Wi-Fi settings, restart the router, reconnect to "
        "the network, and check the network adapter."
    ),
    (
        "keyboard not working",
        "Check the cable or batteries, try another USB port, "
        "and check the keyboard driver."
    ),
    (
        "printer not printing",
        "Check printer power, paper, ink or toner, connection, "
        "and the print queue."
    ),
    (
        "blue screen error",
        "Record the error code, disconnect recently added hardware, "
        "and check drivers and system updates. Back up important data "
        "if possible."
    ),
]

faq_questions = [item[0] for item in faq_data]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq_questions)


@tool
def faq_retrieval(query: str) -> str:
    """Search the hardware FAQ knowledge base for a relevant answer."""

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, faq_vectors)[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score < 0.10:
        return (
            "No sufficiently relevant FAQ was found. "
            "Use the troubleshooting tool or ask the user for more details."
        )

    question, answer = faq_data[best_index]

    return (
        f"Matching FAQ: {question}\n"
        f"Answer: {answer}\n"
        f"Similarity score: {best_score:.2f}"
    )


# =====================================================
# TOOL 3: HARDWARE TROUBLESHOOTING
# =====================================================

@tool
def troubleshooting(issue: str) -> str:
    """Provide step-by-step troubleshooting for hardware problems."""

    issue = issue.lower()

    if any(word in issue for word in [
        "overheat", "overheating", "too hot"
    ]):
        return (
            "1. Shut down the device if it is dangerously hot.\n"
            "2. Keep ventilation openings unobstructed.\n"
            "3. Place the laptop on a hard, flat surface.\n"
            "4. Check whether the cooling fan operates normally.\n"
            "5. If overheating continues, contact a technician."
        )

    elif any(word in issue for word in [
        "not turning on", "won't turn on", "no power"
    ]):
        return (
            "1. Check the power socket and cable.\n"
            "2. Check the charger and battery.\n"
            "3. Disconnect unnecessary peripherals.\n"
            "4. Try powering on again.\n"
            "5. If there is still no power, seek professional repair."
        )

    elif any(word in issue for word in [
        "wifi", "internet", "network"
    ]):
        return (
            "1. Check whether Wi-Fi is enabled.\n"
            "2. Restart the router and computer.\n"
            "3. Reconnect to the correct network.\n"
            "4. Check the network adapter and driver.\n"
            "5. Test another device to isolate the problem."
        )

    elif any(word in issue for word in [
        "keyboard", "mouse", "usb"
    ]):
        return (
            "1. Check the cable, port, or batteries.\n"
            "2. Try another USB port if applicable.\n"
            "3. Test the device on another computer.\n"
            "4. Check its driver and system settings.\n"
            "5. Replace or repair it if necessary."
        )

    elif any(word in issue for word in [
        "slow", "freezing", "lagging", "hang"
    ]):
        return (
            "1. Close unnecessary applications.\n"
            "2. Check CPU and memory usage.\n"
            "3. Check available disk space.\n"
            "4. Restart the computer.\n"
            "5. Check for malware and driver updates."
        )

    else:
        return (
            "1. Identify the exact hardware symptom.\n"
            "2. Check power and physical connections.\n"
            "3. Record any error indicators.\n"
            "4. Test one possible cause at a time.\n"
            "5. Seek professional help if the issue persists."
        )


# =====================================================
# INITIALIZE THE LOCAL LANGUAGE MODEL
# =====================================================

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

print("Local language model initialized.")


# =====================================================
# CREATE THE ReAct AGENT
# =====================================================

tools = [
    calculator,
    faq_retrieval,
    troubleshooting
]

system_prompt = """
You are a ReAct-based technical support agent specializing in
computer and hardware-related questions.

You have three tools:
1. calculator: perform arithmetic calculations.
2. faq_retrieval: search the hardware FAQ knowledge base.
3. troubleshooting: provide practical hardware troubleshooting steps.

Instructions:
- Understand the user's question before selecting a tool.
- Use the appropriate tool when it can help answer the question.
- Use FAQ retrieval for common hardware questions.
- Use troubleshooting for problems requiring diagnostic steps.
- Use the calculator for arithmetic questions.
- You may use multiple tools if necessary.
- Never claim that a tool was used unless it was actually called.
- Give a clear, concise, user-friendly final answer.
- Do not invent hardware specifications or claim that a repair succeeded.
- For electrical hazards, smoke, burning smells, or dangerously hot
  equipment, prioritize safety and recommend shutting down and
  disconnecting power when safe to do so.
"""

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt
)

print("ReAct Technical Support Agent created successfully!")


# =====================================================
# INTERACTIVE QUESTION-ANSWER LOOP
# =====================================================

def ask_support_agent(question: str):
    print("\n" + "=" * 55)
    print("USER QUESTION:", question)
    print("=" * 55)

    try:
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": question}
            ]
        })

        # Display tools actually called by the agent
        for message in result["messages"]:
            if getattr(message, "tool_calls", None):
                for call in message.tool_calls:
                    print("\nTool selected:", call["name"])
                    print("Tool input:", call["args"])

        final_message = result["messages"][-1]

        print("\nTECHNICAL SUPPORT ANSWER:\n")
        print(final_message.content)

    except Exception as error:
        print("\nAgent error:", error)
        print(
            "Check that Ollama is running and that "
            "the qwen2.5:3b model is installed."
        )


# =====================================================
# TEST THE AGENT
# =====================================================

if __name__ == "__main__":
    print("\nAgent is ready!")

    ask_support_agent(
        "My computer is overheating. What should I do?"
    )

    ask_support_agent(
        "My computer is running slowly. How can I fix it?"
    )

    ask_support_agent(
        "Calculate 25 multiplied by 4."
    )

    # Uncomment to interact with the agent:
    #
    # while True:
    #     question = input("\nAsk a hardware question (or 'exit'): ")
    #     if question.lower() == "exit":
    #         break
    #     ask_support_agent(question)
