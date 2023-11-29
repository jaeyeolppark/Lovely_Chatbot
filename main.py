import streamlit as st
from streamlit_chat import message
from langchain.prompts.chat import ChatPromptTemplate
from openai import OpenAI

st.title("연애상담 챗봇")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-1106"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})








# def predict(input, history):
#     history.append({"role":"user", "content":input})
    
#     gpt_response = openai.ChatCompletion.create(
#         model = "gpt-3.5-turbo",
#         messages = history
#     )
    
#     response = gpt_response["choices"][0]["message"]["content"]
#     history.append({"role":"assistant", "content":response})
#     messages = [(history[i]["content"], history[i + 1]["content"]) for i in range(1, len(history), 2)]
#     return messages, history

# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot(label = "ChatBot")
    
#     state = gr.State([{
#         "role" : "system",
#         "content" : "당신은 친절한 상담이 가능한 인공지는 챗봇입니다. 상대방의 말에 공감하고 알맞은 답변을 해주세요."
#     }])
    
#     with gr.Row():
#         txt = gr.Textbox(show_label=False, placeholder="연애관련 고민을 물어보세요")
        
#     txt.submit(predict, [txt, state], [chatbot, state])
    
# demo.launch(debug=True, share = True)

        










# from langchain.chat_models import ChatOpenAI
# chat_model = ChatOpenAI()
# result = chat_model.predict("input_text")
# print(result)


# messages = []
# while True:
#     user_content = input("user : ")
#     messages.append({"role": "user", "content": f"{user_content}"})
#     completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106", messages=messages)
#     assistant_content = completion.choices[0].message["content"].strip()
#     messages.append({"role": "assistant", "content": f"{assistant_content}"})
#     print(f"GPT : {assistant_content}")


# template = "You are a helpful assistant that translates {input_language} to {output_language}."
# human_template = "{text}"

# chat_prompt = ChatPromptTemplate.from_messages([
#     ("system", template),
#     ("human", human_template),
# ])

# chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
