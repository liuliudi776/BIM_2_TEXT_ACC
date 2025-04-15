import requests
from utils.utils_control_logger import control_logger as logger

def send_completion_notification(total_time):
    """
    发送完成手机提醒.
    
    参数:
        total_time: 工作流程的总执行时间
    """
    try:
        base_url = "https://4978.push.ft07.com/send/sctp4978t4dw3rxfmsefc8modxsgh8k.send"
        title = "工作流程完成"
        desp = f"总执行时间: {total_time}"
        # 发送手机提醒，通过 GET 请求调用提醒接口
        response = requests.get(base_url, params={"title": title, "desp": desp})
        if response.status_code != 200:
            logger.warning(f"手机提醒发送失败, 状态码: {response.status_code}")
    except Exception as e:
        logger.warning(f"手机提醒发送时出错: {str(e)}") 