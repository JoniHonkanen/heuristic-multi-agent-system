from schemas import AgentState
from .common import cl, os
import re
import subprocess
import logging  # Voidaan käyttää myöhemmin, jos halutaan siirtää print()-lokit loggingiin


@cl.step(name="Start Docker Container Agent")
async def start_docker_container_agent(state: AgentState):
    print("*** START DOCKER CONTAINER AGENT ***")
    current_step = cl.context.current_step
    current_step.input = "Starting Docker container to execute the generated code."

    os.chdir("generated")
    full_output = ""
    error_output = ""  # Tallennetaan virheiden jäljitys

    try:
        # Rakennetaan Docker image
        print("Building Docker image...")
        build_command = ["docker-compose", "build"]
        with subprocess.Popen(
            build_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        ) as build_process:
            for line in build_process.stdout:
                print(line, end="")
                full_output += line
                await current_step.stream_token(line)
            build_process.wait()
            if build_process.returncode != 0:
                raise Exception("Docker image build failed")

        # Ajetaan Docker container
        print("Running Docker container...")
        up_command = [
            "docker-compose",
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

        state["docker_output"] = full_output
        state["proceed"] = "continue"

    except Exception as e:
        print(f"An error occurred: {e}")
        output_to_use = (
            error_output.strip() if error_output.strip() else full_output.strip()
        )
        state["docker_output"] = f"{str(e)}\n{output_to_use}"
        state["proceed"] = "fix"
        await cl.Message(
            content=f"An error occurred: {e}\nDetails:\n{output_to_use}"
        ).send()

    finally:
        # Siivotaan Docker-resurssit
        subprocess.run(["docker-compose", "down"])
        subprocess.run(["docker", "image", "prune", "-f"])
        os.chdir("..")

    return state
