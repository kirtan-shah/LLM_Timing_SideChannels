import os
import hashlib
from fastapi import FastAPI, HTTPException, Request
from langchain.chat_models import ChatOpenAI
from langchain.cache import GPTCache
from gptcache import Cache, Config
from gptcache.adapter.api import init_similar_cache
from langchain.globals import set_llm_cache
from langchain_openai import OpenAI
from gptcache.similarity_evaluation import OnnxModelEvaluation
from gptcache.embedding import Onnx
import time

# Function to hash LLM name
def get_hashed_name(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()

encoder = Onnx()
onnx_evaluation = OnnxModelEvaluation()
cache_config = Config(similarity_threshold=0.8)

# Function to initialize GPTCache with a hashed directory for each LLM
def init_gptcache(cache_obj: Cache, llm_name: str):
    hashed_llm = get_hashed_name(llm_name)
    #init_similar_cache(cache_obj=cache_obj, data_dir=f"similar_cache_{hashed_llm}") # this line def works
    init_similar_cache(cache_obj=cache_obj, data_dir=f"similar_cache_{hashed_llm}", embedding=encoder, evaluation=onnx_evaluation, config=cache_config)

# Initialize the LLM (GPT-3.5 Turbo)
#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

llm = OpenAI(
    model_name="gpt-3.5-turbo-instruct",
    n=2,
    best_of=2,
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,
)

# Set up GPTCache with LangChain
#cache = Cache()
set_llm_cache(GPTCache(init_gptcache))

# Initialize FastAPI app
app = FastAPI()

@app.on_event("startup")
async def startup():
    print(f"Server is starting... GPTCache with {llm.model_name} is ready!")

save_response = "save"

count = 0

@app.post("/query/")
async def query(request: Request):
    try:
        # Parse user input
        data = await request.json()
        user_input = data.get("input")
        if not user_input:
            raise HTTPException(status_code=400, detail="Input is required.")

        if user_input == "pls flush cache": # for clearing the cache ONLY!
            set_llm_cache(GPTCache(init_gptcache))
            return {"response": "cache flush success"}

        # Query the LLM (uses caching automatically via LangChain)
        #print('hello1')
        start_time = time.time()
        # can add something here from user input that just asks us to flush cache (for our purposes only)
        response = llm.invoke(user_input)
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Query response time: {time_taken:.4f} seconds")
        


        #set_llm_cache(GPTCache(init_gptcache))
        

        return {"response": response}
    except Exception as e:
        # Log the error for debugging
        #print(f"Error: {e}", save_response)
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.get("/")
async def root():
    return {"message": f"GPTCache Server with {llm.model_name} is running!"}


# run server: 
# uvicorn run_chatbot_onnx:app --reload

# post:
""" 
curl -X POST "http://127.0.0.1:8000/query/" \
-H "Content-Type: application/json" \
-d '{"input": "Help me get over my cold."}' 
"""


"""
def init_similar_cache(
    data_dir: str = "api_cache",
    cache_obj: Optional[Cache] = None,
    pre_func: Callable = get_prompt,
    embedding: Optional[BaseEmbedding] = None,
    data_manager: Optional[DataManager] = None,
    evaluation: Optional[SimilarityEvaluation] = None,
    post_func: Callable = temperature_softmax,
    config: Config = Config(),
  ):
  pass

  https://gptcache.readthedocs.io/en/latest/configure_it.html

  https://api.python.langchain.com/en/latest/cache/langchain_community.cache.GPTCache.html


"""