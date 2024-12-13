


run_chatbot_onnx.py - deploy our LangChain application using GPTCache w/ ONNX similarity evaluation model
    to deploy, run: uvicorn run_chatbot_onnx:app --reload

run_chatbot_server.py - deploy our LangChain application using GPTCache w/ default similarity evaluation model
    to deploy, run: uvicorn run_chatbot_server:app --reload

call_query.py - make a query to a server at a specified port
    run: python3 call_query.py --data '{"input": "YOUR INPUT HERE"}'

medquad_attack_with_cache.py - run full pipeline of testing True Label, False Label, and attack sentences for cache hits/misses
    - saves results as a CSV file
    - deploy run_chatbot_onnx server
    - run: python3 medquad_attack_with_cache.py
