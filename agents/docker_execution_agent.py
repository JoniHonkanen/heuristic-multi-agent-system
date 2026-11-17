from schemas import AgentState
from .common import cl, os
import re
import subprocess
import logging  # Voidaan käyttää myöhemmin, jos halutaan siirtää print()-lokit loggingiin


def cap_pulp(requirements: str) -> str:
    """Keep an unpinned PuLP below version 4.

    Generated programs use the PuLP API that models know (LpVariable(lowBound=...),
    PULP_CBC_CMD). PuLP 4 (September 2026, Python >= 3.12) removed both, and it imports numpy
    without depending on it, so an unpinned "PuLP" in a Python 3.12 image now fails at import
    or at the first variable. A version the program pins itself is left as it is.
    """
    lines = requirements.splitlines()
    for i, line in enumerate(lines):
        spec = line.strip()
        name = re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0].lower()
        if name == "pulp" and not re.search(r"==|<|;", spec):
            lines[i] = spec + (",<4" if re.search(r"[>~!]", spec) else "<4")
    return "\n".join(lines) + ("\n" if requirements.endswith("\n") else "")


@cl.step(name="Start Docker Container Agent")
async def start_docker_container_agent(state: AgentState):
    print("*** START DOCKER CONTAINER AGENT ***")
    current_step = cl.context.current_step
    current_step.input = "Starting Docker container to execute the generated code."

    os.chdir("generated")
    full_output = ""
    error_output = ""  # Tallennetaan virheiden jäljitys

    if os.path.exists("requirements.txt"):
        with open("requirements.txt", encoding="utf-8") as f:
            reqs = f.read()
        if cap_pulp(reqs) != reqs:
            with open("requirements.txt", "w", encoding="utf-8") as f:
                f.write(cap_pulp(reqs))

    try:
        # Rakennetaan Docker image
        print("Building Docker image...")
        build_command = ["docker", "compose", "build"]
        with subprocess.Popen(
            build_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        ) as build_process:
            for line in build_process.stdout:
                full_output += line
            build_process.wait()
            if build_process.returncode != 0:
                raise Exception("Docker image build failed")

        # Ajetaan Docker container
        print("Running Docker container...")
        up_command = [
            "docker",
            "compose",
            "up",
            "--abort-on-container-exit",
            "--no-log-prefix",
        ]
        with subprocess.Popen(
            up_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        ) as up_process:
            # Luetaan koko output merkkijonona ja jaetaan rivien mukaan
            output_text = up_process.stdout.read()
            output_lines = output_text.splitlines()
            full_output += "\n" + output_text

        # Prekäännä virhekuviot
        error_patterns = [
            re.compile(r'\s*File\s+".+",\s+line\s+\d+'),  # Python traceback
            re.compile(r"Traceback"),
            re.compile(r"SyntaxError"),
            re.compile(r"ERROR:"),
            re.compile(r"failed to solve:"),
            re.compile(r"exited with code \d+"),
            re.compile(r"http2: server: error"),
        ]

        error_capture = []
        traceback_started = False
        for line in output_lines:
            if any(pattern.search(line) for pattern in error_patterns):
                traceback_started = True
                error_capture.append(line)
            elif traceback_started:
                error_capture.append(line)
            # Lopetetaan, jos virheviesti on loppumassa
            if "exited with code" in line or "failed to solve" in line:
                break

        error_output = "\n".join(error_capture)

        # Tarkistetaan containerin paluuarvo
        if up_process.returncode is not None and up_process.returncode != 0:
            raise Exception("Docker container execution failed")

        # pyritään ratkaisemaan payload liian pitkä -ongelma: otetaan viimeiset 100 riviä ja rajataan ne enintään 4000 merkkiin
        MAX_OUTPUT_CHARS = 4000
        last_100_lines = "\n".join(full_output.strip().splitlines()[-100:])
        trimmed_output = last_100_lines[-MAX_OUTPUT_CHARS:]
        
        print("TRIMMED OUTPUT:")
        print(trimmed_output)
        state["docker_output"] = trimmed_output
        state["proceed"] = "continue"
        current_step.output = trimmed_output 

    except Exception as e:
        print(f"An error occurred: {e}")
        output_to_use = (
            error_output.strip() if error_output.strip() else full_output.strip()
        )
        short_output = "\n".join(output_to_use.splitlines()[-100:])
        state["docker_output"] = f"{str(e)}\n{output_to_use}"
        state["proceed"] = "fix"
        current_step.output = f"{str(e)}\n{short_output}" 
        await cl.Message(
            content=f"An error occurred: {e}\nDetails:\n{short_output}"
        ).send()

    finally:
        # Keep the tail of the unfiltered build + run log; the code fixer can ask for it
        state["docker_full_log"] = "\n".join(full_output.strip().splitlines()[-150:])[-8000:]
        # Siivotaan Docker-resurssit
        subprocess.run(["docker", "compose", "down"])
        subprocess.run(["docker", "image", "prune", "-f"])
        os.chdir("..")

    return state
