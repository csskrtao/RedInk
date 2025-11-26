import os
import re
import base64
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from backend.utils.text_client import get_text_chat_client


class OutlineService:
    def __init__(self):
        print("OutlineService: Initializing...")
        try:
            self.text_config = self._load_text_config()
            print(f"OutlineService: Loaded text config. Active provider: {self.text_config.get('active_provider')}")
            self.client = self._get_client()
            print("OutlineService: Got text chat client.")
            self.prompt_template = self._load_prompt_template()
            print("OutlineService: Loaded prompt template.")
        except Exception as e:
            print(f"OutlineService: Error during initialization: {e}")
            raise # Re-raise to be caught by the outer try-except in api.py

    def _load_text_config(self) -> dict:
        """加载文本生成配置"""
        print("OutlineService: Loading text config...")
        config_path = Path(__file__).parent.parent.parent / 'text_providers.yaml'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                print(f"OutlineService: text_providers.yaml loaded. Config: {config}")
                return config
        # 默认配置
        default_config = {
            'active_provider': 'google_gemini',
            'providers': {
                'google_gemini': {
                    'type': 'google_gemini',
                    'model': 'gemini-2.0-flash-exp',
                    'temperature': 1.0,
                    'max_output_tokens': 65535
                }
            }
        }
        print(f"OutlineService: text_providers.yaml not found, using default config: {default_config}")
        return default_config

    def _get_client(self):
        """根据配置获取客户端"""
        print("OutlineService: Getting text chat client...")
        active_provider = self.text_config.get('active_provider', 'google_gemini')
        providers = self.text_config.get('providers', {})
        provider_config = providers.get(active_provider, {})
        client = get_text_chat_client(provider_config)
        print(f"OutlineService: Client obtained for provider: {active_provider}")
        return client

    def _load_prompt_template(self) -> str:
        print("OutlineService: Loading prompt template...")
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prompts",
            "outline_prompt.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()
            print(f"OutlineService: Prompt template loaded from {prompt_path}")
            return template

    def _parse_outline(self, outline_text: str) -> List[Dict[str, Any]]:
        # 按 <page> 分割页面（兼容旧的 --- 分隔符）
        if '<page>' in outline_text:
            pages_raw = re.split(r'<page>', outline_text, flags=re.IGNORECASE)
        else:
            # 向后兼容：如果没有 <page> 则使用 ---
            pages_raw = outline_text.split("---")

        pages = []

        for index, page_text in enumerate(pages_raw):
            page_text = page_text.strip()
            if not page_text:
                continue

            page_type = "content"
            type_match = re.match(r"\[(\S+)\]", page_text)
            if type_match:
                type_cn = type_match.group(1)
                type_mapping = {
                    "封面": "cover",
                    "内容": "content",
                    "总结": "summary",
                }
                page_type = type_mapping.get(type_cn, "content")

            pages.append({
                "index": index,
                "type": page_type,
                "content": page_text
            })

        return pages

    def generate_outline(
        self,
        topic: str,
        images: Optional[List[bytes]] = None
    ) -> Dict[str, Any]:
        try:
            # print(f"OutlineService.generate_outline called with topic: {topic}, images count: {len(images) if images else 0}") # Removed for more targeted logging
            prompt = self.prompt_template.format(topic=topic)

            if images and len(images) > 0:
                prompt += f"\n\n注意：用户提供了 {len(images)} 张参考图片，请在生成大纲时考虑这些图片的内容和风格。这些图片可能是产品图、个人照片或场景图，请根据图片内容来优化大纲，使生成的内容与图片相关联。"

            # 从配置中获取模型参数
            active_provider = self.text_config.get('active_provider', 'google_gemini')
            providers = self.text_config.get('providers', {})
            provider_config = providers.get(active_provider, {})

            model = provider_config.get('model', 'gemini-2.0-flash-exp')
            temperature = provider_config.get('temperature', 1.0)
            max_output_tokens = provider_config.get('max_output_tokens', 65535)

            outline_text = self.client.generate_text(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                images=images
            )
            # print(f"generate_text returned: {outline_text[:200]}...") # Removed for more targeted logging

            pages = self._parse_outline(outline_text)

            return {
                "success": True,
                "outline": outline_text,
                "pages": pages,
                "has_images": images is not None and len(images) > 0
            }

        except Exception as e:
            error_msg = str(e)
            # print(f"Error in OutlineService.generate_outline: {error_msg}") # Removed for more targeted logging
            raise e # Re-raise to be caught by the outer try-except in api.py for better error visibility


def get_outline_service() -> OutlineService:
    """
    获取大纲生成服务实例
    每次调用都创建新实例以确保配置是最新的
    """
    return OutlineService()
