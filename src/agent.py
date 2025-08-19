import logging

import modal

logger = logging.getLogger("agent")

def download_files():
    import subprocess
    subprocess.run(["uv", "run", "src/agent.py", "download-files"], cwd="/root")


image = (
    modal.Image.debian_slim()

    .uv_pip_install(
        "fastapi>=0.116.1",
        "livekit-agents[openai,turn-detector,silero,cartesia,deepgram]~=1.2",
        "livekit-plugins-noise-cancellation~=0.2",
        "modal>=1.1.2",
    )
)

app = modal.App("livekit-example", image=image)

# Create a persisted dict - the data gets retained between app runs
room_dict = modal.Dict.from_name("room-dict", create_if_missing=True)

with image.imports():
    import asyncio

    from fastapi import FastAPI, Request, Response
    from livekit import api
    from livekit.agents import (
        NOT_GIVEN,
        Agent,
        AgentFalseInterruptionEvent,
        AgentSession,
        JobContext,
        JobProcess,
        MetricsCollectedEvent,
        RoomInputOptions,
        RunContext,
        WorkerOptions,
        cli,
        metrics,
    )
    from livekit.agents.llm import function_tool
    from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
    from livekit.plugins.turn_detector.multilingual import MultilingualModel


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location.

        If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.

        Args:
            location: The location to look up weather information for (e.g. city name)
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all providers at https://docs.livekit.io/agents/integrations/llm/
        llm=openai.LLM(model="gpt-4o-mini"),
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all providers at https://docs.livekit.io/agents/integrations/stt/
        stt=deepgram.STT(model="nova-3", language="multi"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all providers at https://docs.livekit.io/agents/integrations/tts/
        tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead:
    # session = AgentSession(
    #     # See all providers at https://docs.livekit.io/agents/integrations/realtime/
    #     llm=openai.realtime.RealtimeModel()
    # )

    # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
    # when it's detected, you may resume the agent's speech
    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/integrations/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/integrations/avatar/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()

@app.cls(
    timeout=3000, 
    secrets=[modal.Secret.from_name("livekit-voice-agent")],
    enable_memory_snapshot=True,
    min_containers=1,
)
@modal.concurrent(max_inputs=10)
class LiveKitAgentServer:

    @modal.enter(snap=True)
    def enter(self):
        import subprocess
        subprocess.run(["uv", "run", "src/agent.py", "download-files"], cwd="/root")

    @modal.enter(snap=False)
    def start_agent_server(self):
        import subprocess
        import threading
        def run_dev():
            subprocess.run(["uv", "run", "src/agent.py", "dev"], cwd="/root")
        thread = threading.Thread(target=run_dev, daemon=True)
        thread.start()

    @modal.asgi_app()
    def webhook_app(self):

        web_app = FastAPI()

        @web_app.post("/")
        async def webhook(request: Request):

            token_verifier = api.TokenVerifier()
            webhook_receiver = api.WebhookReceiver(token_verifier)

            auth_token = request.headers.get("Authorization")
            if not auth_token:
                return Response(status_code=401)

            body = await request.body()
            event = webhook_receiver.receive(body.decode("utf-8"), auth_token)
            print("received event:", event)

            room_name = event.room.name
            event_type = event.event

            # ## check whether the room is already in the room_dict
            if room_name in room_dict and event_type == "room_started":
                print(
                    f"Received web event for room {room_name} that already has a worker running"
                )
                return

            if event_type == "room_started":
                room_dict[room_name] = True
                print(f"Worker for room {room_name} spawned")
                while room_dict[room_name]:
                    await asyncio.sleep(1)

                del room_dict[room_name]

            elif event_type in ["room_finished", "participant_left"]:
                if room_name in room_dict and room_dict[room_name]:
                    room_dict[room_name] = False
                    print(f"Worker for room {room_name} spun down")
                elif room_name not in room_dict:
                    print(f"Worker for room {room_name} not found")
                elif room_name in room_dict and not room_dict[room_name]:
                    print(f"Worker for room {room_name} already spun down")

            return Response(status_code=200)

        return web_app


if __name__ == "__main__":

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
