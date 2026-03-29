from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端
    LLM 调用的抽象层, 换模型不用改逻辑, Agent 统一调用
    """
    def __init__(self, model:str, api_key:str, base_url:str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt:str, system_prompt:str) -> str:
        """
        调用LLM API来生成回应
        """
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role':'system', 'content': system_prompt},
                {'role':'user', 'content':prompt}
            ]
            response = self.client.chat.completions.create(
                model = self.model,
                messages=messages,  # type: ignore
                stream=False    
            )
            # 从返回里取出模型说的话
            answer = response.choices[0].message.content or ""
            # 把模型的回答交给上层（Agent / UI）
            print("大语言模型响应成功。")
            return answer   # type: ignore
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"