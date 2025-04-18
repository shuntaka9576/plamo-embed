import { pipeline, env, AutoTokenizer, AutoModel } from '@huggingface/transformers';

// env.localModelPath = 'models/';
// 
// env.allowRemoteModels = false;

const modelId = 'pfnet/plamo-embedding-1b'
const tokenizer = await AutoTokenizer.from_pretrained(modelId)

// const classifier = await pipeline('text-generation', modelId);
// 
// const result = await classifier('こんにちは！');

// console.log(result);




/*
env.localModelPath = 'models/';

env.allowRemoteModels = false;

const query = "PLaMo-Embedding-1Bとは何ですか？"
const documents = [
    "PLaMo-Embedding-1Bは、Preferred Networks, Inc. によって開発された日本語テキスト埋め込みモデルです。",
    "最近は随分と暖かくなりましたね。",
]
const modelId = 'pfnet/plamo-embedding-1b'

// @ts-ignore trust_remote_code is required for PLaMo models
const tokenizer = await AutoTokenizer.from_pretrained(modelId, { trust_remote_code: true })
// @ts-ignore trust_remote_code is required for PLaMo models
const model = await AutoModel.from_pretrained(modelId, { trust_remote_code: true })

// @ts-ignore PLaMo custom methods
const query_embedding = model.encode_query(query, tokenizer)
// @ts-ignore PLaMo custom methods
const document_embeddings = model.encode_document(documents, tokenizer)

/*
class MyClassificationPipeline {
  static task = 'text-classification';
  static model = 'pfnet/plamo-embedding-1b';
  static instance = null;

  static async getInstance(progress_callback = null) {
    if (this.instance === null) {
      // NOTE: Uncomment this to change the cache directory
      // env.cacheDir = './.cache';

      this.instance = pipeline(this.task, this.model, { progress_callback });
    }

    return this.instance;
  }
}
*/
