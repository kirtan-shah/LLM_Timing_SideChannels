import os
import subprocess
import requests
import time
import argparse
import json

parser = argparse.ArgumentParser(description="Parse command-line arguments.")

parser.add_argument("--data", type=str, required=True, help="Your prompt")
args = parser.parse_args()
print(args.data)

url = "http://127.0.0.1:8000/query/"
header = "Content-Type: application/json"
#data = args.data
curl_command = f'curl -X POST "{url}" -H "{header}" -d \'{args.data}\''
print(curl_command)

start_time = time.time()

# Execute the curl command and capture the output
result = subprocess.run(
    ["curl", "-X", "POST", url, "-H", header, "-d", args.data],
    text=True,  # Ensures output is returned as a string (not bytes)
    capture_output=True  # Captures stdout and stderr
)

end_time = time.time()
time_taken = end_time - start_time

parsed_json = json.loads(result.stdout)

# Access the output
print("Output:", parsed_json["response"])
print(f"Client-side query response time: {time_taken:.4f} seconds")
#print("Error:", result.stderr)
#print("Exit Code:", result.returncode)

# python3 call_query.py --data '{"input": "should i wear a jacket when its cold"}'

#Develop a meeting outline for the interdisciplinary team to review and discuss the treatment plan for the patient Maria diagnosed with Natural short sleeper.

# python3 call_query.py --data '{"input": "covid symptoms"}'
# python3 call_query.py --data '{"input": "what are common covid symptoms?"}'