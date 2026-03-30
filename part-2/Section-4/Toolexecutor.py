from typing import Dict, Any
from Tool import search

class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    给这个类创建一个叫 tools 的属性，专门存所有可用的工具
    tools = {
    "工具名": { 工具的详细信息 },
    "search": { 函数、描述、参数... },
    "draw": { ... }
}
    """
    def __init__(self):
        self.tools:Dict[str,Dict[str,Any]] = {}

    def registerTool(self, name:str, description:str, func:callable):
        """
        向工具箱中注册一个新工具。
        """

        if name in self.tools:
            print(f"警告：工具'{name}'已存在，将被覆盖。")
        self.tools[name] = {"description":description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name:str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name,{}).get("func")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}" 
            # items() 作用：把字典里的 键值对 一对一对拿出来
            # info = 一个字典,info["description"] → 工具描述,info["func"] → 工具函数
            for name, info in self.tools.items()
        ])

if __name__ == "__main__":
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search",search_description,search)

    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")
