from openai import OpenAI
import numpy as np
import time
import requests

def flush():
    requests.post("http://localhost:30000/flush_cache")
    # print("Flushing cache")

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

flush()

def timed_completion(prompt):
    start_time = time.perf_counter()
    completion = client.completions.create(model="TheBloke/Llama-2-7B-GPTQ", #"facebook/opt-125m",
                prompt=prompt, temperature=.7, max_tokens=1)
    end_time = time.perf_counter()
    return completion.choices[0].text, (end_time - start_time)*1000

N = 1000
times = []
for _ in range(N):
    flush()
    time.sleep(.2)
    _, ms = timed_completion("Python is a ")
    times.append(ms)
times = np.array(times)
avg_time = np.mean(times)
std_time = np.sqrt(np.var(times))
print(f"N={N}, Average time w/flush: \t\t{avg_time:.3f} ms, sigma: {std_time}")

times = []
for _ in range(N):
    _, ms = timed_completion("Python is a ")
    time.sleep(.2)
    times.append(ms)
times = np.array(times)
avg_time_2 = np.mean(times)
std_time_2 = np.sqrt(np.var(times))
print(f"N={N}, Average time: \t\t\t{avg_time_2:.3f} ms, sigma: {std_time_2}")
sigma_sum = np.sqrt(std_time*std_time + std_time_2*std_time_2)
print(f"Timing difference: \t\t\t{(avg_time - avg_time_2):.3f} ms, sigma: {sigma_sum:.3f}")
# curl http://localhost:30000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#         "model": "TheBloke/Llama-2-7B-GPTQ",
#         "prompt": "San Francisco is a",
#         "max_tokens": 1,
#         "temperature": 0
#     }'
