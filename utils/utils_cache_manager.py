import hashlib
from diskcache import Cache
from utils.utils_control_logger import control_logger as logger

def _clear_cache_for_prompts(prompts, config):
    """清除指定提示的缓存"""
    cache = Cache("./openai_cache")
    
    for prompt in prompts:
        cache_key = hashlib.md5(
            f"{prompt}_{config['current_model']['type']}".encode()
        ).hexdigest()
        if cache_key in cache:
            cache.delete(cache_key)
            logger.debug(f"已删除缓存: {cache_key[:8]}...")