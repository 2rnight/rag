# test.py
import httpx

response = httpx.post(
    "https://api.siliconflow.cn/v1/embeddings",
    headers={
        "Authorization": "Bearer sk-bvmyxjwswbammqxwcubjzibhbzigpfpustlajyjzgivhqtnd",
        "Content-Type": "application/json"
    },
    json={
        "model": "BAAI/bge-m3",
        "input": "测试文本"
    },
    timeout=30
)

print("状态码:", response.status_code)
print("是否成功:", "embedding" in response.text)
print("响应:", response.text[:100])