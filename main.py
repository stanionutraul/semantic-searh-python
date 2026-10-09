
import ast
import json
import math
import operator
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

OPERATIONS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculator(expression):
    """Executa expresii matematice simple, fara cod arbitrar."""

    if not isinstance(expression, str) or len(expression) > 100:
        raise ValueError("Expresia este invalida sau prea lunga.")

    tree = ast.parse(expression, mode="eval")
    nodes = list(ast.walk(tree))

    if len(nodes) > 50:
        raise ValueError("Expresia este prea complexa.")

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                return node.value
            raise ValueError("Sunt permise doar numere.")

        if isinstance(node, ast.BinOp):
            operation = OPERATIONS.get(type(node.op))

            if operation is None:
                raise ValueError("Operatie nepermisa.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponentul este prea mare.")

            result = operation(left, right)

        elif isinstance(node, ast.UnaryOp):
            operation = OPERATIONS.get(type(node.op))

            if operation is None:
                raise ValueError("Operatie nepermisa.")

            result = operation(evaluate(node.operand))

        else:
            raise ValueError("Expresie matematica neacceptata.")

        if isinstance(result, (int, float)):
            if abs(result) > 1e100:
                raise ValueError("Rezultatul este prea mare.")
            if not math.isfinite(result):
                raise ValueError("Rezultatul nu este finit.")

        return result

    return evaluate(tree.body)


def ask_ollama(messages, json_mode=False):
    """Trimite mesajele catre modelul Ollama."""

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }

    if json_mode:
        payload["format"] = "json"

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    return response.json()["message"]["content"]


def ask_agent(question):
    """Decide daca foloseste calculatorul sau raspunde direct."""

    instructions = """
Esti un AI Agent care poate folosi un calculator.

Trebuie sa raspunzi exclusiv cu un obiect JSON valid.

Pentru calcule, raspunde astfel:
{"tool": "calculator", "expression": "124 * 32"}

IMPORTANT:
- Pune expresia matematica originala in campul expression.
- Nu calcula tu rezultatul.
- Nu inlocui expression cu answer.

Pentru intrebari care nu necesita calcule, raspunde astfel:
{"tool": "none", "answer": "raspunsul pentru utilizator"}

Nu adauga text in afara obiectului JSON.
"""

    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": question},
    ]

    try:
        content = ask_ollama(messages, json_mode=True)
        decision = json.loads(content)

        if not isinstance(decision, dict):
            return "Eroare: modelul nu a returnat un obiect JSON."

        tool = decision.get("tool")

        if tool == "calculator":
            expression = decision.get("expression")

            if not isinstance(expression, str):
                return (
                    "Eroare: modelul a ales calculatorul, "
                    "dar nu a furnizat expresia matematica."
                )

            result = calculator(expression)

            messages.append({
                "role": "assistant",
                "content": content,
            })
            messages.append({
                "role": "user",
                "content": (
                    f"Calculatorul a executat expresia "
                    f"{expression!r} si a returnat {result}. "
                    "Explica rezultatul utilizatorului. "
                    "Nu modifica rezultatul calculului."
                ),
            })

            return ask_ollama(messages)

        if tool == "none":
            answer = decision.get("answer")

            if isinstance(answer, str) and answer.strip():
                return answer

            return "Eroare: modelul nu a furnizat un raspuns."

        return (
            "Eroare: decizie necunoscuta. "
            "Modelul trebuie sa aleaga calculator sau none."
        )

    except requests.RequestException as error:
        return (
            f"Eroare de conexiune cu Ollama: {error}\n"
            "Verifica daca Ollama ruleaza."
        )
    except (ValueError, KeyError, TypeError) as error:
        return f"Eroare la procesarea raspunsului: {error}"
    except Exception as error:
        return f"Eroare neasteptata: {error}"


def main():
    print("=" * 35)
    print("       AI AGENT WITH TOOLS")
    print("=" * 35)
    print(f"Model: {MODEL}")
    print("Instrument disponibil: calculator")
    print("Scrie 'exit' pentru a iesi.\n")

    while True:
        try:
            question = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nLa revedere!")
            break

        if question.lower() in ("exit", "quit", "iesire"):
            print("La revedere!")
            break

        if not question:
            continue

        answer = ask_agent(question)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()