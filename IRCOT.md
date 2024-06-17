## Usages
https://www.51cto.com/aigc/922.html
```sh
$ python -m venv ./myenv
$ source ./myenv/bin/activate
$ deactivate
$ pip install -r requirements.txt
$ pip install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

DATA=sample
LLM=qwen
SYNONYM_THRESH=0.8
GPUS=0
LLM_API=openai
https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890 OPENAI_API_KEY=sk-cyKVhwtgX6g2aiVGA8C55eAa771749628fAbD0A9075214F8 OPENAI_BASE_URL=http://112.28.49.224:18000/v1 bash src/setup_hipporag_colbert.sh $DATA $LLM $GPUS $SYNONYM_THRESH $LLM_API

data=sample
extraction_model=qwen
available_gpus=0
syn_thresh=0.8
llm_api=openai
extraction_type=ner

OPENAI_API_KEY=sk-cyKVhwtgX6g2aiVGA8C55eAa771749628fAbD0A9075214F8 OPENAI_BASE_URL=http://112.28.49.224:18000/v1 python src/openie_with_retrieval_option_parallel.py --dataset $data --llm $llm_api --model_name $extraction_model --run_ner --num_passages all

OPENAI_API_KEY=sk-cyKVhwtgX6g2aiVGA8C55eAa771749628fAbD0A9075214F8 OPENAI_BASE_URL=http://112.28.49.224:18000/v1 python src/named_entity_extraction_parallel.py --dataset $data --llm $llm_api --model_name $extraction_model

DATA=sample
LLM=qwen
SYNONYM_THRESH=0.8
GPUS=0
LLM_API=openai
RETRIEVER=colbertv2
OPENAI_API_KEY=sk-cyKVhwtgX6g2aiVGA8C55eAa771749628fAbD0A9075214F8 OPENAI_BASE_URL=http://112.28.49.224:18000/v1 python src/ircot_hipporag.py --dataset $DATA --retriever $RETRIEVER --llm $LLM_API --llm_model $LLM --max_steps 3 --doc_ensemble f --top_k 10  --sim_threshold $SYNONYM_THRESH --damping 0.5
```

## IRCOT Instruction
```
You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".

您作为智能助手，擅长帮助用户在多个文档之间进行复杂的多跳推理。这项任务通过演示来说明，每个演示包括一组文档和相关的问题以及多跳推理思考。您的任务是为当前步骤生成一个思考，不要一次生成所有思考！如果您认为已经到达最终步骤，请以“所以答案是：”开始。
```
