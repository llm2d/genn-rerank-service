from datetime import datetime
from fastapi import FastAPI, Request
import logging
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import uvicorn
from pydantic import BaseModel
import os
import numpy as np


from fastapi import FastAPI, Security, HTTPException


from typing import List, Optional, Any
from FlagEmbedding import FlagReranker
from pydantic import BaseModel

app = FastAPI()
security = HTTPBearer()

# constants
# RERANK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/maidalun1020/bce-reranker-base_v1")
RERANK_MODEL_PATH = "maidalun1020/bce-reranker-base_v1"
env_bearer_token = "genn"


# schema
class Response(BaseModel):
    code: int
    message: str
    data: Any

    class Config:
    # 指定 code 字段在 JSON Schema 中的类型为 integer
        schema_extra = {
            "properties": {
                "code": {"type": "integer"}
            }
        }


class Inputs(BaseModel):
    id: str
    text: Optional[str]


class QADocs(BaseModel):
    query: Optional[str]
    inputs: Optional[List[Inputs]]


# service
class Singleton(type):
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class Reranker(metaclass=Singleton):
    def __init__(self, model_path):
        self.reranker = FlagReranker(model_path, use_fp16=False)

    def compute_score(self, pairs: List[List[str]]):
        if len(pairs) > 0:
            result = self.reranker.compute_score(pairs)
            if isinstance(result, float):
                result = [result]
            return result
        else:
            return None


class Chat(object):
    def __init__(self, rerank_model_path: str = RERANK_MODEL_PATH):
        self.reranker = Reranker(rerank_model_path)

    def fit_query_answer_rerank(self, query_docs: QADocs) -> List:
        if query_docs is None or len(query_docs.inputs) == 0:
            return []
        new_docs = []
        pair = []
        for answer in query_docs.inputs:
            pair.append([query_docs.query, answer.text])
        scores = self.reranker.compute_score(pair)
        for index, score in enumerate(scores):
            new_docs.append(
                {
                    "id": query_docs.inputs[index].id,
                    "text": query_docs.inputs[index].text,
                    "score": 1 / (1 + np.exp(-score))}
            )
        new_docs = list(sorted(new_docs, key=lambda x: x["score"], reverse=True))
        return new_docs

# logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(levelname)s: %(asctime)s - %(filename)s:%(lineno)d - %(message)s"
)


@app.middleware("http")
async def measure_response_time(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    end_time = datetime.now()
    response_time_ms = (end_time - start_time).total_seconds() * 1000
    response.headers["X-Response-Time"] = f"{response_time_ms:.2f} ms"
    return response


@app.get(path="/ping", response_model=Response)
async def handle_ping_request():
    return Response(code=0, message="pong", data=None)


@app.post(path="/rerank", response_model=Response)
async def handle_rerank_request(
    docs: QADocs, credentials: HTTPAuthorizationCredentials = Security(security)
):
    logger.info(f"rerank request: {docs}")
    token = credentials.credentials
    if env_bearer_token is not None and token != env_bearer_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    chat = Chat()
    qa_docs_with_rerank = chat.fit_query_answer_rerank(docs)
    logger.info(f"rerank response: {qa_docs_with_rerank}")
    return Response(code=0, message="rerank successful", data=qa_docs_with_rerank)


if __name__ == "__main__":
    token = os.getenv("ACCESS_TOKEN")
    if token is not None:
        env_bearer_token = token
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"api start failed!\error\n{e}")
