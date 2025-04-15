import hashlib
import asyncio
from functools import wraps
from typing import List, Union
from diskcache import Cache
from openai import AsyncOpenAI
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.console import Console 
import json
import tiktoken
import os
import datetime
from utils.utils_control_logger import control_logger as logger

from utils.utils_gpt_logger import gpt_logger

# ----------------------------------------------------------------------
# 全局变量和初始化
# ----------------------------------------------------------------------
# 创建缓存实例
cache = Cache("./openai_cache")
# 全局 token 计数器
_total_tokens = 0
_input_tokens = 0     # 输入 token 数量
_output_tokens = 0    # 输出 token 数量
waiting_count = 0     # 全局等待计数器

# 速率限制相关的全局变量（针对不同事件循环的锁和上次请求时间）
_rate_limit_locks = {}
_last_request_times = {}

# 响应时间统计，用于记录每个部署的统计数据
_deployment_stats = {}

# 退避策略相关变量
_base_delay = 1   # 基础延迟时间（秒）
_max_delay = 60   # 最大延迟时间（秒）

# ----------------------------------------------------------------------
# 部署统计类
# ----------------------------------------------------------------------
class DeploymentStats:
    def __init__(self):
        self.total_time = 0       # 总响应时间（秒）
        self.request_count = 0    # 请求总次数
        self.total_tokens = 0     # 总 tokens 数

    @property
    def average_tokens_per_second(self):
        return self.total_tokens / self.total_time if self.total_time > 0 else 0

    def add_response_time(self, time,total_tokens):
        self.total_time += time
        self.request_count += 1
        self.total_tokens += total_tokens

# ----------------------------------------------------------------------
# 速率控制函数（动态延迟）
# ----------------------------------------------------------------------
async def rate_limit():
    """
    速率控制函数：每次请求前检查上次请求时间，若不足 _base_delay 则等待。
    """
    delay = _base_delay
    loop = asyncio.get_running_loop()
    if loop not in _rate_limit_locks:
        _rate_limit_locks[loop] = asyncio.Lock()
        _last_request_times[loop] = loop.time()
    
    async with _rate_limit_locks[loop]:
        now = loop.time()
        last_request_time = _last_request_times[loop]
        elapsed = now - last_request_time
        wait_time = delay - elapsed
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        _last_request_times[loop] = loop.time()

# ----------------------------------------------------------------------
# 装饰器：缓存 OpenAI API 响应
# ----------------------------------------------------------------------
def async_cache_openai_response(func):
    """OpenAI API 响应的缓存装饰器"""
    @wraps(func)
    async def wrapper(prompt: Union[str, List[str]], config):
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        results = []
        for single_prompt in prompts:
            cache_key = hashlib.md5(
                f"{single_prompt}_{config['current_model']['type']}".encode()
            ).hexdigest()
            
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                gpt_logger.trace(
                    f"\n{'='*100}\n"
                    f"🎯 命中缓存\n"
                    f"{'='*100}\n"
                    f"提示词: {single_prompt}\n"
                    f"缓存结果:\n{cached_result}\n"
                    f"{'='*100}\n"
                )
                results.append(cached_result)
                continue
                
            result = await func(single_prompt, config)
            cache.set(cache_key, result)
            gpt_logger.trace(
                f"\n{'='*100}\n"
                f"💫 缓存新结果\n"
                f"{'='*100}\n"
                f"提示词: {single_prompt[:10]+'...' if len(single_prompt) > 10 else single_prompt}\n" \
                f"缓存内容:\n{result[:10]+'...' if len(result) > 10 else result}\n" \
                f"{'='*100}\n"
            )
            results.append(result)
            
        return results[0] if isinstance(prompt, str) else results
    return wrapper

# ----------------------------------------------------------------------
# 模型选择与轮询
# ----------------------------------------------------------------------
_current_model_index = 0
def get_next_model(config):
    """
    根据当前模型类型，在配置中选取下一个可用模型
    """
    global _current_model_index
    current_type = config['current_model']['type']
    same_type_models = [
        model for model in sorted(config['gpt_config']['models'], key=lambda x: x['priority'])
        if model['type'] == current_type
    ]
    
    if not same_type_models:
        raise Exception(f"没有找到类型为 {current_type} 的可用模型。")
    
    _current_model_index = (_current_model_index + 1) % len(same_type_models)
    return same_type_models[_current_model_index]

# ----------------------------------------------------------------------
# 异步调用 OpenAI API（支持超时重试和退避策略）
# ----------------------------------------------------------------------
@async_cache_openai_response
async def async_call_openai_api(prompt: str, config, max_retries: int = 15, initial_timeout: float = 120):
    """
    异步调用 OpenAI API 进行文本处理，支持多端点轮询和递增超时重试
    """
    global _total_tokens, _input_tokens, _output_tokens, waiting_count
    
    current_delay = _base_delay
    
    for retry_count in range(max_retries):
        await rate_limit()
        
        current_timeout = min(initial_timeout * (1.5 ** retry_count), 240)
        current_model = get_next_model(config)
        config['current_model'] = current_model
        deployment = current_model["deployment"]
        
        if deployment not in _deployment_stats:
            _deployment_stats[deployment] = DeploymentStats()
        
        client = AsyncOpenAI(
            api_key=current_model["api_key"], 
            base_url=current_model["api_base"]
        )

        waiting_count += 1
        start_time = asyncio.get_event_loop().time()
        try:
            # 记录发出请求内容
            gpt_logger.trace(
                f"\n{'='*100}\n"
                f"🔄 OpenAI API调用\n"
                f"{'='*100}\n"
                f"提示词: {prompt}\n"
                f"模型类型: {current_model['type']}\n"
                f"部署: {deployment}\n"
                f"请求超时: {current_timeout:.2f}秒\n"
                f"{'='*100}\n"
            )

            completion = await asyncio.wait_for(
                client.chat.completions.create(
                    model=current_model["model"],
                    messages=[
                        {"role": "system", "content": "你是土木工程领域专家"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                ),
                timeout=current_timeout
            )
            
            # 成功后重置延迟
            current_delay = _base_delay
            
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time
            _deployment_stats[deployment].add_response_time(response_time, completion.usage.total_tokens)
            
            response = completion.choices[0].message.content.strip()
            _total_tokens += completion.usage.total_tokens
            _input_tokens += completion.usage.prompt_tokens
            _output_tokens += completion.usage.completion_tokens
            
            gpt_logger.trace(
                f"\n{'='*100}\n"
                f"📝 OpenAI API调用详情\n"
                f"{'='*100}\n"
                f"提示词: {prompt}\n"
                f"模型类型: {current_model['type']}\n"
                f"部署: {deployment}\n"
                f"响应时间: {response_time:.2f}秒\n"
                f"平均token输出速度: {_deployment_stats[deployment].average_tokens_per_second:.2f} tokens/秒\n"
                f"响应内容:\n{response}\n"
                f"本次使用tokens: {completion.usage.total_tokens}\n"
                f"总计使用tokens: {_total_tokens}\n"
                f"输入tokens: {_input_tokens}, 输出tokens: {_output_tokens}\n"
                f"{'='*100}\n"
            )
            return response
            
        except (asyncio.TimeoutError, json.decoder.JSONDecodeError) as e:
            error_type = "JSON解码错误" if isinstance(e, json.decoder.JSONDecodeError) else "超时"
            gpt_logger.trace(
                f"\n{'='*100}\n"
                f"⚠️ OpenAI API调用{error_type}\n"
                f"部署: {deployment}\n"
                f"重试次数: {retry_count + 1}/{max_retries}\n"
                f"错误信息: {str(e)}\n"
                f"{'='*100}\n"
            )
            if retry_count == max_retries - 1:
                raise Exception(f"OpenAI API在{max_retries}次尝试后仍然失败: {str(e)}")
            continue
            
        except Exception as e:
            error_str = str(e).lower()
            if ("429" in str(e) or 
                "rate limit" in error_str or 
                "504" in str(e) or 
                "500" in str(e) or
                # "502" in str(e) or
                "503" in str(e) or
                "gateway timeout" in error_str):
                
                current_delay = min(current_delay * 2, _max_delay)
                gpt_logger.trace(
                    f"\n{'='*100}\n"
                    f"⚠️ 遇到限制或超时\n"
                    f"部署: {deployment}\n"
                    f"错误类型: {'速率限制' if '429' in str(e) else '服务器错误' if '500' in str(e) else 'Gateway超时'}\n"
                    f"等待时间: {current_delay}秒\n"
                    f"重试次数: {retry_count + 1}/{max_retries}\n"
                    f"prompt: {prompt}\n"
                    f"{'='*100}\n"
                )
                if retry_count > 10:
                    gpt_logger.warning(
                        f"⚠️ 遇到限制或超时,重试超过10次\n"
                        f"部署: {deployment}\n"
                        f"等待时间: {current_delay}秒\n"
                        f"{'='*100}\n"
                    )
                await asyncio.sleep(current_delay)
                if retry_count < max_retries - 1:
                    continue

            gpt_logger.error(
                f"\n{'='*100}\n"
                f"❌ OpenAI API调用失败\n"
                f"部署: {deployment}\n"
                f"错误信息: {str(e)}\n"
                f"{'='*100}\n"
            )
            raise
            
        finally:
            waiting_count -= 1
            await client.close()

# ----------------------------------------------------------------------
# 批量处理多个提示词（使用任务池+实时进度条）
# ----------------------------------------------------------------------
def batch_get_cache(prompts: List[str], config) -> dict:
    """
    批量获取缓存结果
    
    Args:
        prompts: 提示词列表
        config: 配置信息
    
    Returns:
        dict: {cache_key: cached_result} 格式的字典
    """
    # 批量生成所有cache keys
    cache_keys = [
        hashlib.md5(
            f"{prompt}_{config['current_model']['type']}".encode()
        ).hexdigest()
        for prompt in prompts
    ]
    
    # 从磁盘缓存读取到内存
    cache_results = {}
    for key in cache_keys:
        result = cache.get(key)
        if result is not None:
            cache_results[key] = result
            
    return cache_results

async def batch_process_prompts(prompts: List[str], config, batch_size: int = 100, enforce_model_type: str = "default"):
    """
    批量处理多个提示词，维持固定数量的并发任务，并使用 Rich 显示实时进度条
    """
    # batch_size = 5 # 批处理强制锁定为5
    if enforce_model_type != "default":
        # 记录当前模型类型
        current_model_type_res = config['current_model']['type']
        config['current_model']['type'] = enforce_model_type

    results = [None] * len(prompts)
    tasks = set()
    next_prompt_index = 0
    global waiting_count
    
    # 首先批量获取缓存结果
    prompt_to_key = {
        prompt: hashlib.md5(
            f"{prompt}_{config['current_model']['type']}".encode()
        ).hexdigest()
        for prompt in prompts
    }
    key_to_prompt = {v: k for k, v in prompt_to_key.items()}
    
    # 批量获取缓存结果
    cache_results = batch_get_cache(prompts, config)
    
    # 处理缓存命中的结果
    prompts_to_process = []
    for i, prompt in enumerate(prompts):
        cache_key = prompt_to_key[prompt]
        if cache_key in cache_results:
            results[i] = cache_results[cache_key]
            gpt_logger.trace(
                f"\n{'='*100}\n"
                f"🎯 命中缓存\n"
                f"{'='*100}\n"
                f"提示词: {prompt}\n"
                f"缓存结果:\n{cache_results[cache_key]}\n"
                f"{'='*100}\n"
            )
        else:
            prompts_to_process.append((i, prompt))

    def get_cost_description():
        input_price = 2.5
        output_price = 10

        """生成成本和预估成本描述"""
        input_cost = (_input_tokens / 1_000_000) * input_price
        output_cost = (_output_tokens / 1_000_000) * output_price
        current_cost = input_cost + output_cost
        
        # 使用 tiktoken 计算预估 token
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # 计算所有prompts的预估token数
        total_estimated_tokens = sum(len(encoding.encode(p)) for p in prompts)
        # 假设输出token等于输入token
        total_cost = (total_estimated_tokens * 1) * (input_price / 1_000_000)  # 输入成本
        total_cost += (total_estimated_tokens * 1) * (output_price / 1_000_000)*0.1  # 输出成本
        
        return (f"\n当前成本: ¥{current_cost:.2f} 元"
                f" | 预估总成本: ¥{total_cost:.2f} 元")
    
    def get_stats_description():
        """生成包含各部署平均token输出速度的描述"""
        stats = []
        for deployment, stat in _deployment_stats.items():
            if stat.request_count > 0:
                stats.append(f"{deployment}: {stat.average_tokens_per_second:.2f} tokens/s")
        cost_desc = get_cost_description()
        return "\n".join(stats) + f"  {cost_desc}"
    
    async def process_single_prompt(prompt, index, progress, task_id):
        try:
            result = await async_call_openai_api(prompt, config)
            results[index] = result
            
            # 将新结果写入缓存
            cache_key = prompt_to_key[prompt]
            cache.set(cache_key, result)
            
            # 更新进度条描述
            stats_desc = get_stats_description()
            current_model_type = config['current_model']['type']
            new_description = (
                f"prompts处理中 [模型: {current_model_type}] [等待响应: {waiting_count}] [任务池: {len(tasks)-1}/{batch_size}] [总tokens: {_total_tokens}]\n"
                f"{stats_desc}"
            )
            progress.update(
                task_id,
                description=new_description,
                advance=1
            )
            return result
        except Exception as e:
            gpt_logger.error(f"处理提示词 {index} 失败: {str(e)}")
            raise

    # 只处理未命中缓存的提示词
    if prompts_to_process:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
        ) as progress:
            current_model_type = config['current_model']['type']
            task_id = progress.add_task(
                description=f"prompts处理中 [模型: {current_model_type}] [等待响应: {waiting_count}] [任务池: 0/{batch_size}] [总tokens: {_total_tokens}]",
                total=len(prompts_to_process)
            )
            
            prompt_index = 0
            while prompt_index < len(prompts_to_process) and len(tasks) < batch_size:
                idx, prompt = prompts_to_process[prompt_index]
                task = asyncio.create_task(
                    process_single_prompt(prompt, idx, progress, task_id)
                )
                tasks.add(task)
                progress.update(
                    task_id,
                    description=f"prompts处理中 [模型: {current_model_type}] [等待响应: {waiting_count}] [任务池: {len(tasks)}/{batch_size}] [总tokens: {_total_tokens}]"
                )
                prompt_index += 1
            
            while tasks:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    await task
                while prompt_index < len(prompts_to_process) and len(tasks) < batch_size:
                    idx, prompt = prompts_to_process[prompt_index]
                    task = asyncio.create_task(
                        process_single_prompt(prompt, idx, progress, task_id)
                    )
                    tasks.add(task)
                    progress.update(
                        task_id,
                        description=f"prompts处理中 [等待响应: {waiting_count}] [任务池: {len(tasks)}/{batch_size}] [总tokens: {_total_tokens}]"
                    )
                    prompt_index += 1

    display_deployment_stats()
    # 处理完成后，恢复默认模型类型
    if enforce_model_type != "default":
        config['current_model']['type'] = current_model_type_res
    return results

# ----------------------------------------------------------------------
# 使用 Rich Table 显示各部署的统计信息
# ----------------------------------------------------------------------
def display_deployment_stats():
    """
    使用 Rich Table 显示各个部署的统计信息：
    展示部署、请求次数、响应时间、总 tokens 数以及平均 tokens/秒等数据，
    表格底部显示总 token 使用量及成本信息。
    """
    console = Console()
    table = Table(title="部署统计信息", show_lines=True)

    table.add_column("部署", style="cyan", no_wrap=True)
    table.add_column("请求次数", justify="center", style="green")
    table.add_column("响应时间 (秒)", justify="center", style="magenta")
    table.add_column("平均tokens/s", justify="center", style="red")

    for deployment, stat in _deployment_stats.items():
        table.add_row(
            deployment,
            str(stat.request_count),
            f"{stat.total_time:.2f}",
            f"{stat.average_tokens_per_second:.2f}"
        )

    input_cost = (_input_tokens / 1_000_000) * 2
    output_cost = (_output_tokens / 1_000_000) * 8
    table.caption = f"总Tokens: {_total_tokens} | 成本: 输入 ¥{input_cost:.2f}, 输出 ¥{output_cost:.2f}"

    console.print(table)

async def process_single_prompt_special(prompt: str, config):
    """
    特殊处理单个提示词，不使用进度条
    """
    # 生成缓存键
    cache_key = hashlib.md5(
        f"{prompt}_{config['current_model']['type']}".encode()
    ).hexdigest()
    
    # 检查缓存
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        logger.trace(f"处理提示词: {prompt} 命中缓存，结果已返回。")
        return cached_result

    try:
        result = await async_call_openai_api(prompt, config)
        
        # 将新结果写入缓存
        cache.set(cache_key, result)
        
        logger.trace(f"处理提示词: {prompt} 成功，结果已缓存。")
        return result
    except Exception as e:
        logger.error(f"处理提示词 {prompt} 失败: {str(e)}")
        raise