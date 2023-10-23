import os
import re
import json
import openai
from copy import deepcopy

import chainlit as cl
from chainlit.input_widget import Slider
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from src.utils import parse_config_and_return_vectordb_dict

os.environ["OPENAI_API_KEY"] = open("openai_api.txt", "r").readline()
openai.api_key = os.environ.get("OPENAI_API_KEY")

model_name = "gpt-3.5-turbo"
vectordb_config = {
    "knowledge_path_list":[
        ["data/clean_card_data/overview_creditcard.txt"],
        ["data/clean_card_data/cubcardlist.txt", "data/clean_card_data/cube卡.txt"],
        ["data/clean_card_data/shopee.txt"],
        ["data/clean_card_data/eva.txt"],
        ["data/clean_card_data/world.txt"],
    ],
    "db_path":["vectordb/campaign", 
               "vectordb/cube",
               "vectordb/shopee",
               "vectordb/eva",
               "vectordb/world",],
    "db_name_list":["Campaign", "Cube", "Shopee", "Eva", "World"]
}

text_splitter = CharacterTextSplitter(separator = "\n", chunk_size=1, chunk_overlap=0)
embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base", model_kwargs={'device': 'cuda'})
vectordb_dict = parse_config_and_return_vectordb_dict(vectordb_config["knowledge_path_list"], vectordb_config["db_path"], vectordb_config["db_name_list"], embeddings, text_splitter)

@cl.on_settings_update
async def setup_agent(settings):
    pass

@cl.set_chat_profiles
async def chat_profile(current_user: cl.AppUser):
    return [
        cl.ChatProfile(
            name="Campaign",
            markdown_description="全卡限時行銷活動",
        ),
        cl.ChatProfile(
            name="Cube",
            markdown_description="Cube卡權益",
        ),
        cl.ChatProfile(
            name="Shopee",
            markdown_description="蝦皮卡權益",
        ),
        cl.ChatProfile(
            name="Eva",
            markdown_description="長榮航空聯名卡權益",
        ),
        cl.ChatProfile(
            name="World",
            markdown_description="世界卡權益",
        ),
    ]

@cl.action_callback(name="answer_regenerate_actions")
async def on_regenerate_actions(action):
    if action.value == "answer_regenerate":
        chat_settings = cl.user_session.get("chat_settings")
        message_history = cl.user_session.get("message_history")
        # message_history.pop()

        content = message_history[-1]["content"]

        regenerate_answer =  cl.Message(author="重新生成", content="", language="str", indent=1)
        async for stream_resp in await openai.ChatCompletion.acreate(
            model=model_name,
            messages= message_history + [{"role": "user", "content": content}],
            stream=True,
            temperature = 1.0,
            max_tokens = int(chat_settings["max_tokens"]),
        ):
            token = stream_resp.choices[0]["delta"].get("content", "")
            await regenerate_answer.stream_token(token)

       # message_history.append({"role": "assistant", "content": regenerate_answer.content})

        cl.user_session.set("regenerate_answer", regenerate_answer)
        await regenerate_answer.send()

    else:
        cl.ErrorMessage(content="Invalid action").send()
    

@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "你是國泰世華的聊天機器人-阿發, [檢索資料]是由國泰世華提供的。參考[檢索資料]使用中文簡潔和專業的回覆顧客的問題, 如果答案不在公開資料中, 請說 “對不起, 我所擁有的公開資料中沒有相關資訊, 請您換個問題或將問題描述得更詳細, 讓阿發能正確完整的回答您”，不允許在答案中加入編造的內容。\n[檢索資料]\n{knowledge}\n\n"}],
    )
    settings = await cl.ChatSettings(
        [
            Slider(
                id="temperature",
                label="創意程度",
                initial=0,
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id="max_tokens",
                label="最大字數",
                initial=1024,
                min=512,
                max=2048,
                step=128,
            ),
        ]
    ).send()

@cl.on_message
async def main(message: str):
    message_history = cl.user_session.get("message_history")
    chat_settings = cl.user_session.get("chat_settings")
    chat_profile = cl.user_session.get("chat_profile")
    vectordb = vectordb_dict[chat_profile]

    knowledge = "\n".join([doc[0].page_content for doc in vectordb.similarity_search_with_score(message.content, 5)])
    message_history[0]["content"] = re.sub(r'\{[^\}]*\}', '{' + knowledge + '}', message_history[0]["content"])
    message_history.append({"role": "user", "content": message.content})
    await cl.Message(
            author=chat_profile + "知識庫",
            content=knowledge,
            language="str",
            indent=1
        ).send()
    
    current_answer = cl.user_session.get("current_answer")
    if current_answer:
        await current_answer.remove_actions()

    answer_regenerate_actions = [cl.Action(name="answer_regenerate_actions", label="重新生成", value="answer_regenerate", description="")]
    current_answer = cl.Message(author="阿發",content="", actions=answer_regenerate_actions)
    async for stream_resp in await openai.ChatCompletion.acreate(
        model=model_name,
        messages= message_history + [{"role": "user", "content": message.content}],
        stream=True,
        temperature = float(chat_settings["temperature"]),
        max_tokens = int(chat_settings["max_tokens"]),
    ):
        token = stream_resp.choices[0]["delta"].get("content", "")
        await current_answer.stream_token(token)

    message_history.append({"role": "assistant", "content": current_answer.content})

    cl.user_session.set("current_answer", current_answer)
    await current_answer.send()



    
    