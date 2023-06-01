import os, tempfile, ssl, openai, re
from faster_whisper import WhisperModel
from aiohttp import web
from dotenv import load_dotenv


### Load variables with keys - file would be .env
load_dotenv()

# Create an SSL context
ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain('ldn1whisper.crt', 'ldn1whisper.key')

# OpenAI keys
openai.api_type = "azure"
openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
openai.api_version = "2023-05-15"
openai.api_key = os.getenv('AZURE_OPENAI_KEY')

model = WhisperModel('large-v2', device="cuda", compute_type="float32")  ### Change to int8 or int8_float16 if you have a more capable GPU (AWS)



async def transcribe(request):
    try:
        reader = await request.multipart()
        field = await reader.next()

        temp_file_descriptor, temp_filepath = tempfile.mkstemp(suffix='.wav')

        with open(temp_filepath, 'wb') as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)

        segments, info = model.transcribe(temp_filepath, beam_size=5, best_of=5)

        transcription = ''
        for segment in segments:
            transcription += f"{segment.txt}\n"



        return web.json_response({"transcription":  transcription})

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def gpt_process(transcript):

    try:

        transcript = await transcript.json()
        transcript = transcript['transcript']

        ### Set up the system prompts to instruct our bot how to behave. This happens before any questions are submitted
        messages = []
        messages.append({"role": "system","content": "Categorize the call transcription as 'closure' if it relates to closure. Otherwise, categorize it as 'other'.\n\nEvaluate the agent's performance on the following error types and factors only for calls categorized as 'closure':\n\nClosures\n- Originating Account not closed, or closed at a later time\n- Non-SE closure applied when should have been SE\n- If SE - No information has been provided on RG organizations and tools\n- Applied Incorrect Closure Duration for SE which is less than required\n\nLinked\n- Accounts that should have been closed remain open\n- Accounts that could have been located using the minimum 4-step guide have not been closed where appropriate\n\nEffective Questioning\n- No use of Effective Questioning takes place before a decision was made\n- No use of effective questioning (closure reason & investigated further) when member requested to close\n\nProcess\n- Subject Access Request process not followed correctly\n- Namediff not asked when required\n\nBalance\n- Failing to return balance above £1 on Primary Account (SE CLOSURE)\n- Failing to return balance when specifically requested by the player\n\nNotes\n- Notes do not correspond to what was discussed or have missing/unclear key information\n- Noting CCC when comfy/control has been confirmed\n- Notes do not reference any RG concerns raised\n- Closure duration not noted if confirmed to player\n\nDPA\n- Not asking DPA at all (sensitive/account information discussed)\n- Incorrect or unsure answers accepted where inappropriate\n- Information disclosed to a third party\n\nFor each factor, assign a pass or fail based on the call transcript. Provide a brief explanation of why a pass or fail was given for each factor.\n\nEvaluate the agent's performance on QA by analyzing the call transcription for sentiment and QA issues.\n\nPlease provide a summary and a TL;DR summary.\n\nInclude tags for categorization.\n\nElaborate on why a pass or fail was given for each factor and QA evaluation.\n"})

        messages.append({"role": "assistant","content": "Categorize the call transcription as 'closure' if it relates to closure. Otherwise, categorize it as 'other'.\n\nEvaluate the agent's performance on the following error types and factors only for calls categorized as 'closure':\n\nClosures\n- Originating Account not closed, or closed at a later time\n- Non-SE closure applied when should have been SE\n- If SE - No information has been provided on RG organizations and tools\n- Applied Incorrect Closure Duration for SE which is less than required\n\nLinked\n- Accounts that should have been closed remain open\n- Accounts that could have been located using the minimum 4-step guide have not been closed where appropriate\n\nEffective Questioning\n- No use of Effective Questioning takes place before a decision was made\n- No use of effective questioning (closure reason & investigated further) when member requested to close\n\nProcess\n- Subject Access Request process not followed correctly\n- Namediff not asked when required\n\nBalance\n- Failing to return balance above £1 on Primary Account (SE CLOSURE)\n- Failing to return balance when specifically requested by the player\n\nNotes\n- Notes do not correspond to what was discussed or have missing/unclear key information\n- Noting CCC when comfy/control has been confirmed\n- Notes do not reference any RG concerns raised\n- Closure duration not noted if confirmed to player\n\nDPA\n- Not asking DPA at all (sensitive/account information discussed)\n- Incorrect or unsure answers accepted where inappropriate\n- Information disclosed to a third party\n\nFor each factor, assign a pass or fail based on the call transcript. Provide a brief explanation of why a pass or fail was given for each factor.\n\nEvaluate the agent's performance on QA by analyzing the call transcription for sentiment and QA issues.\n\nPlease provide a summary and a TL;DR summary.\n\nInclude tags for categorization.\n\nElaborate on why a pass or fail was given for each factor and QA evaluation.\n"})


        messages.append({"role": "user", "content": f"{transcript}"})  # Add the chat history to the messages after the system prompts

        # Call the OpenAI API to generate a chat completion
        response = openai.ChatCompletion.create(   
            engine="eit_gptturbo",   
            messages=messages     
        )

        # Get the generated text from the API response
        text_output = response["choices"][0]["message"]["content"]

        return web.json_response({"summary": text_output}, status=200)

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
    
    
async def index(request):
    return web.FileResponse('templates/index.html')



app = web.Application()
app.router.add_get('/', index)
app.router.add_post('/transcribe', transcribe)
app.router.add_post('/talkwithme', gpt_process)


web.run_app(app, host='0.0.0.0', port=5000, ssl_context=ssl_context)
