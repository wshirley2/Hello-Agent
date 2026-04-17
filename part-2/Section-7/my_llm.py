# 从 hello_agents 库中导入 HelloAgentsLLM 基类，然后创建一个名为 MyLLM 的新类继承它。
# 如何通过继承 HelloAgentsLLM，来增加对 ModelScope 平台的支持

import os
from typing import Optional
from openai import OpenAI
from hello_agents import HelloAgentsLLM

class MyLLM(HelloAgentsLLM):
    """
    自定义LLM：完美支持 魔搭(ModelScope) + AIHubmix
    两个平台完全隔离，互不冲突
    """
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        # ==========================================
        # 模式1：专用 AIHubmix
        # ==========================================
        if provider == "aihubmix":
            # 强制读取AIHubmix独立配置
            aihubmix_key = os.getenv("AIHubmix_API_KEY")
            aihubmix_model = os.getenv("AIHubmix_MODEL_ID")
            aihubmix_url = os.getenv("AIHubmix_BASE_URL")

            if not aihubmix_key:
                raise ValueError("AIHubmix_API_KEY 未配置")
            
            # 直接调用父类，走标准OpenAI协议
            super().__init__(
                model=aihubmix_model,
                api_key=aihubmix_key,
                base_url=aihubmix_url,
                provider="openai",
                **kwargs
            )
            return

        # ==========================================
        # 模式2：专用 魔搭(ModelScope)
        # ==========================================
        elif provider == "modelscope":
            super().__init__(
                model=model or os.getenv("LLM_MODEL_ID"),
                api_key=api_key or os.getenv("LLM_API_KEY"),
                base_url="https://api-inference.modelscope.cn/v1/",
                provider="modelscope",
                **kwargs
            )
            return

        # ==========================================
        # 默认模式：自动识别
        # ==========================================
        else:
            super().__init__(
                model=model, api_key=api_key, base_url=base_url,
                provider=provider, **kwargs
            )
            
            