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

## Sample Workflow

## Running Open Information Extraction

#### 1. Load ./data/sample_corpus.json
- document['passage'] = document['title'] + '\n' + document['text']

#### 2. Export ./output/openie_sample_results_*.json

##### NER(Named Entity Recognition) Prompt

Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.

Paragraph: 
```
Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.
```

{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}

Paragraph:
```
{user_input}
```

##### OpenIE(Open Information Extraction) Prompt
Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""
```

{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}

{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"]
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}

Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
古巴位于加勒比海西北部、墨西哥湾入口，优越的地理位置和气候条件，使古巴成为热门旅游目的地。古巴旅游部数据显示，截至4月26日，古巴年内接待国际游客突破100万人次。古巴政府预计2024年将接待国际游客350万人次。中国游客方面，2010年至2019年，赴古巴的中国游客数量年均增长约23%。
```

{named_entity_json}
[
  "古巴",
  "加勒比海",
  "西北部",
  "墨西哥湾",
  "入口",
  "旅游部",
  "4月26日",
  "2024年",
  "中国游客",
  "2010年",
  "2019年"
]

{
  "named_entities": [
    "古巴",
    "加勒比海",
    "墨西哥湾",
    "古巴旅游部",
    "4月26日",
    "2024年",
    "中国游客",
    "2010年",
    "2019年"
  ],
  "triples": [
    [
      "古巴",
      "位于",
      "加勒比海"
    ],
    [
      "古巴",
      "位于",
      "墨西哥湾入口"
    ],
    [
      "古巴",
      "是",
      "热门旅游目的地"
    ],
    [
      "古巴旅游部",
      "发布数据",
      "截至4月26日"
    ],
    [
      "古巴",
      "接待国际游客",
      "100万人次"
    ],
    [
      "2024年",
      "预计接待国际游客",
      "350万人次"
    ],
    [
      "中国游客",
      "数量增长",
      "23%"
    ],
    [
      "2010年",
      "至",
      "2019年"
    ],
    [
      "中国游客",
      "赴",
      "古巴"
    ]
  ]
}

#### 3. Export ./output/sample_queries.named_entity_output.tsv & Load ./data/sample.json

##### NER(Named Entity Recognition) Prompts
You're a very effective entity extraction system.

Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?
{"named_entities": ["First for Women", "Arthur's Magazine"]}

Question: {user_input}

## Creating ColBERT Graph

#### 1. 
- possible_files: ./output/openie_sample_results_ner_qwen_*.json 
- extracted_file: ./output/openie_sample_results_ner_qwen_3.json
- passage_json: ./output/sample_facts_and_sim_graph_passage_chatgpt_openIE.ents_only_lower_preprocess_ner_qwen.v3.subset.json
- node_json: ./output/sample_facts_and_sim_graph_nodes_chatgpt_openIE.ents_only_lower_preprocess_ner_qwen.v3.subset.json
- fact_json: ./output/sample_facts_and_sim_graph_clean_facts_chatgpt_openIE.ents_only_lower_preprocess_ner_qwen.v3.subset.json
