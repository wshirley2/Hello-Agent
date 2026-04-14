"""
智能搜索助手 - 基于 LangGraph + Tavily API 的真实搜索系统
1. 理解用户需求
2. 使用Tavily API真实搜索信息  
3. 生成基于搜索结果的回答
"""

import asyncio
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

import os
from dotenv import load_dotenv
from tavily import TavilyClient

from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver

# 加载环境变量
load_dotenv()
"""
一个关键的设计是同时包含了 user_query 和 search_query 字段。
这允许智能体先将用户的自然语言提问，优化成更适合搜索引擎的精炼关键词，
从而显著提升搜索结果的质量。
"""
# 1. 定义全局状态
# 为状态对象定义了一个清晰的数据模式（Schema）
class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str # 经过LLM理解后的用户需求总结
    search_query: str # 优化后用于Tavily API的搜索查询
    search_results: str# Tavily搜索返回的结果
    final_answer: str # 最终生成的答案
    step: str # 标记当前步骤

# 初始化模型和Tavily客户端
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.7
)

# 初始化Tavily客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# 2. 定义工作流节点
# 2.1 理解与查询节点
"""此节点是工作流的第一步，此节点的职责是理解用户意图，并为其生成一个最优化的搜索查询。"""

def understand_query_node(state: SearchState) -> dict:
    """步骤1: 理解用户查询并生成搜索关键词"""
    # user_message = state["message"][-1].content # state["message"] → 整个对话列表, [-1] = 取列表里最后一条消息, 从工作流状态里取最后一条消息内容

    # 获取最新的用户消息
    user_message = ""
    for msg in reversed(state["messages"]):
        # 判断这条是不是【用户说的话】,isinstance = 判断类型, HumanMessage = 用户发的消息
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    understand_prompt = f"""分析用户的查询："{user_message}"
    请完成两个任务：
    1. 简洁总结用户想要了解什么
    2. 生成最适合搜索引擎的关键词（中英文均可，要精准）

    格式：
    理解：[用户需求总结]
    搜索词：[最佳搜索关键词]"""

    # invoke 是调用 AI、发送请求、获取回复的方法，SystemMessage系统消息告诉 AI：你要扮演什么角色、要做什么任务
    response = llm.invoke([
        SystemMessage(content="你是一个擅长理解用户问题并生成搜索关键词的助手"),
        HumanMessage(content=understand_prompt)
        ])
    response_text = response.content

    # 解析LLM的输出，提取搜索关键词
    search_query = user_message # 默认使用原始查询
    if "搜索词：" in response_text:
        search_query = response_text.split("搜索词：")[1].strip()
    elif "搜索关键词：" in response_text:
        search_query = response_text.split("搜索关键词：")[1].strip()

    """该节点通过一个结构化的提示，要求 LLM 同时完成“意图理解”和“关键词生成”两个任务，并将解析出的专用搜索关键词更新到状态的 search_query 字段中，为下一步的精确搜索做好准备。"""
    # 打包返回给工作流框架，我理解了什么，我要搜什么，现在走到哪一步，要对用户说什么
    return{
        "user_query": response_text,
        "search_query": search_query,
        "step": "understood", # step = 当前节点执行到哪一步, "understood" = 已理解用户意图
        "messages": [AIMessage(content=f"我将为您搜索：{search_query}")] # 给用户一个明确反馈，告诉用户 AI 要干嘛了 
    }

# 2.2 搜索节点
def tavily_search_node(state: SearchState) -> dict:
    """此节点通过 tavily_client.search 发起真实的 API 调用。它被包裹在 try...except 块中，用于捕获可能的异常。如果搜索失败，它会更新 step 状态为 "search_failed"，这个状态将被下一个节点用来触发备用方案。"""

    """步骤2: 使用Tavily API进行真实搜索"""
    search_query = state["search_query"]
    try:
        print(f"🔍 正在搜索: {search_query}")
        # 调用Tavily搜索API
        response = tavily_client.search(
            query=search_query,
            search_depth="basic",
            max_results=5,      # 最多返回几条结果
            include_answer=True # 是否直接返回AI总结答案
        )

        # 处理搜索结果
        search_results = ""

        # 优先使用Tavily的综合答案, 检查搜索结果里有没有 AI 总结好的答案
        if response.get("answer"):
            search_results = f"综合答案：\n{response['answer']}\n\n"

        # 添加具体的搜索结果, 检查有没有搜到具体网页
        if response.get("results"):
            search_results += "相关信息：\n"
            # 给每条结果编号：1、2、3, 只取前 3 条最相关的结果, 每条结果提取 3 个信息
            for i, result in enumerate(response["results"][:3], 1):
                title = result.get("title", "")     # 标题
                content = result.get("content", "") # 内容摘要
                url = result.get("url", "")         # 来源链接
                search_results += f"{i}. {title}\n{content}\n来源：{url}\n\n"
        
        if not search_results:
            search_results = "抱歉，没有找到相关信息。"
        
        return {
            "search_results" : search_results,
            "step": "searched",
            "messages": [AIMessage(content=f"✅ 搜索完成！找到了相关信息，正在为您整理答案...")]
        }
    
    except Exception as e:
        error_msg = f"搜索时发生错误: {str(e)}"
        print(f"❌ {error_msg}")

        return {
            "search_results": f"搜索失败：{error_msg}",
            "step": "search_failed",
            "messages": [AIMessage(content="❌ 搜索遇到问题，我将基于已有知识为您回答")]
        }

# 2.3 回答节点
def generate_answer_node(state: SearchState) -> dict:
    """步骤3: 基于搜索结果生成最终答案"""

    if state["step"] == "search_failed":
        # 如果搜索失败，执行回退策略，基于LLM自身知识回答
        fallback_prompt = f"搜索API暂时不可用，请基于您的知识回答用户的问题：\n用户问题：{state['user_query']}"
        response = llm.invoke([
            SystemMessage(content="你是一个知识问答助手"),
            HumanMessage(content=fallback_prompt)
        ])

        return {
            "final_answer": response.content,
            "step": "completed",
            "messages": [AIMessage(content=response.content)]
        }
    
    else:
        # 若搜索成功，基于搜索结果生成答案
        answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：
        用户问题：{state['user_query']}
        搜索结果：\n{state['search_results']}

        请要求：
        1. 综合搜索结果，提供准确、有用的回答
        2. 如果是技术问题，提供具体的解决方案或代码
        3. 引用重要信息的来源
        4. 回答要结构清晰、易于理解
        5. 如果搜索结果不够完整，请说明并提供补充建议"""

        response = llm.invoke([
            SystemMessage(content="你是一个基于搜索结果回答问题的助手"),
            HumanMessage(content=answer_prompt)
        ])

    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)]
    }
    
# 构建搜索工作流,即构建图
def creat_search_assistant():
    # 创建一个工作流（图）, StateGraph = 工作流 / 流程图框架（来自 LangGraph）
    workflow = StateGraph(SearchState)

    # 添加节点
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)

    # 设置线性流程，链接边
    workflow.add_edge(START,"understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
    
    # 开始 → 理解问题 → 搜索 → 生成答案 → 结束
    # START：流程起点  END：流程终点
    
    # 编译图
    memory = InMemorySaver() # 内存记忆，记录对话历史
    # 一个封装好的 AI 助手实例(编译好的工作流对象),它里面装好了：你定义的流程图（理解 → 搜索 → 回答）,记忆功能,所有节点函数,执行逻辑
    # compile = 把你画的流程图 → 变成 可直接运行的 “可执行程序 / 机器人”
    # 不 compile：只是图纸，不能运行，compile 后：变成 app 对象，可以直接用
    app = workflow.compile(checkpointer=memory) # 把流程图编译成可运行的程序
    return app # 返回这个做好的 AI 助手程序


async def main():
    """主函数：运行智能搜索助手"""

    # 检查API密钥
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ 错误：请在.env文件中配置TAVILY_API_KEY")
        return
    
    app = creat_search_assistant()

    print("🔍 智能搜索助手启动！")
    print("我会使用Tavily API为您搜索最新、最准确的信息")
    print("支持各种问题：新闻、技术、知识问答等")
    print("(输入 'quit' 退出)\n")

    session_count = 0

    # 无限循环 → 可以一直提问，不会问一次就退出
    while True:
        user_input = input("🤔 您想了解什么: ").strip()

        if user_input.lower() in ['quit', 'q', '退出', 'exit']:
            print("感谢使用！再见！👋")
            break

        if not user_input:
            continue

        # 给每一次提问，创建一个独立的 会话 ID, 让 LangGraph 知道：这是一次新的提问，不要和上一次混淆
        session_count += 1
        # thread_id = 对话线程 ID, 每次都不一样, 第一次:search-session-1
        config = {"configurable":{"thread_id": f"search-session-{session_count}"}}

        # 初始状态
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": "",
            "search_query": "",
            "search_results": "",
            "final_answer": "",
            "step": "start"
        }

        try:
            print("\n" + "="*60)

            # 执行工作流, astream可以看到中间过程、可以实时打印进度、可以流式输出
            # app.astream() 会返回：一步一步的【节点执行结果】，执行完一个节点，就返回一个结果，是流式、分段返回的
            # {
            #     "节点名字": 这个节点return回来的所有数据
            # }
            async for output in app.astream(initial_state, config=config):
                # 当前正在运行的节点名字 and 这个节点返回的数据
                for node_name, node_output in output.items():
                    if "messages" in node_output and node_output["messages"]:
                        latest_message = node_output["messages"][-1]
                        if isinstance(latest_message, AIMessage):
                            if node_name == "understand":
                                print(f"🧠 理解阶段: {latest_message.content}")
                            elif node_name == "search":
                                print(f"🔍 搜索阶段: {latest_message.content}")
                            elif node_name == "answer":
                                print(f"\n💡 最终回答:\n{latest_message.content}")
            print("\n" + "="*60 + "\n")

        except Exception as e:
            print(f"❌ 发生错误: {e}")
            print("请重新输入您的问题。\n")

if __name__ == "__main__":
    asyncio.run(main())