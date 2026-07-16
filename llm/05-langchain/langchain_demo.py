"""LangChain: LCEL chain, memory, tool, agent"""
import os, math
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent

def get_llm():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("OPENAI_API_KEY gerekli")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=key)

def lcel_chain(llm):
    print("=" * 50)
    print("  LCEL CHAIN (| operatörü)")
    print("=" * 50)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sen Türkçe yanıt veren bir asistansın. Her yanıtı 1-2 cümleyle ver."),
        ("user", "{soru}"),
    ])
    chain = prompt | llm | StrOutputParser()
    yanit = chain.invoke({"soru": "Python'da dictionary nedir?"})
    print(f"  {yanit}\n")

def memory_demo(llm):
    print("=" * 50)
    print("  MEMORY (sohbet geçmişi)")
    print("=" * 50)

    history = []

    sorular = [
        "Benim adım Ahmet.",
        "En sevdiğim renk mavi.",
        "Benim adım ne demiştim?",
        "En sevdiğim renk neydi?",
    ]

    for soru in sorular:
        msgs = [("system", "Kısa yanıt ver. Sohbet geçmişini hatırla.")]
        for h in history:
            msgs.append(h)
        msgs.append(("user", soru))

        prompt = ChatPromptTemplate.from_messages(msgs)
        chain = prompt | llm | StrOutputParser()
        yanit = chain.invoke({})
        history.append(("user", soru))
        history.append(("ai", yanit))
        print(f"  👤 {soru}")
        print(f"  🤖 {yanit}\n")

def tool_demo(llm):
    print("=" * 50)
    print("  TOOL KULLANIMI")
    print("=" * 50)

    @tool
    def hesap_makinesi(islem: str) -> str:
        """Matematiksel ifadeyi hesaplar. Örn: '25 * 4 + 10'"""
        try:
            sonuc = eval(islem, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi, "sin": math.sin, "cos": math.cos, "abs": abs, "round": round, "pow": pow})
            return f"Sonuç: {sonuc}"
        except:
            return "Hesaplanamadı"

    @tool
    def kelime_say(metin: str) -> str:
        """Verilen metindeki kelime sayısını döndürür."""
        return f"Kelime sayısı: {len(metin.split())}"

    tools = [hesap_makinesi, kelime_say]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sen araç kullanabilen bir asistansın."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    sorular = [
        "125 × 37 kaç eder?",
        "'Python ile yapay zeka öğrenmek çok keyifli' cümlesinde kaç kelime var?",
    ]
    for soru in sorular:
        yanit = executor.invoke({"input": soru})
        print(f"  👤 {soru}")
        print(f"  🤖 {yanit['output']}\n")

if __name__ == "__main__":
    llm = get_llm()
    lcel_chain(llm)
    memory_demo(llm)
    tool_demo(llm)
