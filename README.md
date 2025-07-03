# Change the Channel: Timing Side Channel Exploits for LLM Server Caches
Authors: Kirtan Shah and Madurya Suresh

## Paper
[paper.pdf](paper.pdf)

## Presentation
[presentation.pdf](presentation.pdf)

## File Structure
`KVCacheTiming/TimingMeasurement.ipynb`:
- first run `KVCacheTiming/launch_server.sh` for the SGLang LLM server
- execute notebook to generate visualizations of cache hit/miss classifier ROC curve and ping timings

`run_chatbot_onnx.py` - deploy our LangChain application using GPTCache w/ ONNX similarity evaluation model
    to deploy, run: uvicorn run_chatbot_onnx:app --reload

`run_chatbot_server.py` - deploy our LangChain application using GPTCache w/ default similarity evaluation model
    to deploy, run: uvicorn run_chatbot_server:app --reload

`call_query.py` - make a query to a server at a specified port
    run: python3 call_query.py --data '{"input": "YOUR INPUT HERE"}'

`medquad_attack_with_cache.py` - run full pipeline of testing True Label, False Label, and attack sentences for cache hits/misses
    - saves results as a CSV file
    - deploy run_chatbot_onnx server
    - run: python3 medquad_attack_with_cache.py

`GeneralAttack/attack/attack_interface.py` -  the generalized Peeping Neighbor Attack for any prompt template.


`GeneralAttack/attack/MedQuadAttack.ipynb` - runs attack on medical question prompt

`GeneralAttack/attack/HRDatasetAttack.ipynb` - runs attack on HR salary prompt

`CacheHitMiss.csv` - data from running medquad_attack_with_cache.py. 
- Dataset Number: which semantic dataset the sample comes from 
- Label Type: True Label (predicted to be semantically similar) or False Label (predicted to be semantically different)
- Attack Sentence Number: 0 is best attack sentence --> 5 is worst/least representative of a large semantic space
- Query Response Time: time between attacker's query and receiving a response
- Same Answer: whether or not attacker and victim received the same response string

`get_cache_attack_data.py` - gather some data/visualizations for the presentation/report
