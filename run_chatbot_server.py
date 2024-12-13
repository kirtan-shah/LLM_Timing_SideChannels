import os
import hashlib
from fastapi import FastAPI, HTTPException, Request
from langchain.chat_models import ChatOpenAI
from langchain.cache import GPTCache
from gptcache import Cache
from gptcache.adapter.api import init_similar_cache
from langchain.globals import set_llm_cache
from langchain_openai import OpenAI
from gptcache.similarity_evaluation import OnnxModelEvaluation, SearchDistanceEvaluation
from gptcache.embedding import Onnx
import time
from gptcache.similarity_evaluation.exact_match import ExactMatchEvaluation
from gptcache.similarity_evaluation.np import NumpyNormEvaluation

# Function to hash LLM name
def get_hashed_name(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()

encoder = Onnx()
onnx_evaluation = OnnxModelEvaluation()

# Function to initialize GPTCache with a hashed directory for each LLM
def init_gptcache(cache_obj: Cache, llm_name: str):
    hashed_llm = get_hashed_name(llm_name)
    #init_similar_cache(cache_obj=cache_obj, data_dir=f"similar_cache_{hashed_llm}", evaluation=NumpyNormEvaluation())
    init_similar_cache(cache_obj=cache_obj, data_dir=f"similar_cache_{hashed_llm}") # this line def works
    #init_similar_cache(cache_obj=cache_obj, data_dir=f"similar_cache_{hashed_llm}", embedding=encoder, evaluation=onnx_evaluation)

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
cache = Cache()
set_llm_cache(GPTCache(init_gptcache))

# Initialize FastAPI app
app = FastAPI()

@app.on_event("startup")
async def startup():
    print(f"Server is starting... GPTCache with {llm.model_name} is ready!")

save_response = "save"

@app.post("/query/")
async def query(request: Request):
    try:
        # Parse user input
        data = await request.json()
        user_input = data.get("input")
        if not user_input:
            raise HTTPException(status_code=400, detail="Input is required.")

        # Query the LLM (uses caching automatically via LangChain)
        #print('hello1')
        start_time = time.time()
        response = llm.invoke(user_input)
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Query response time: {time_taken:.4f} seconds")
        #save_response = response
        #print('hello')
        #print(response)

        return {"response": response}
    except Exception as e:
        # Log the error for debugging
        #print(f"Error: {e}", save_response)
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.get("/")
async def root():
    return {"message": f"GPTCache Server with {llm.model_name} is running!"}


