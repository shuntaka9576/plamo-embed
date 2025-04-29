import { PGlite } from '@electric-sql/pglite';
import { NodeFS } from '@electric-sql/pglite/nodefs';
import { vector } from "@electric-sql/pglite/vector";

const pglite = new PGlite({
    fs: new NodeFS("./db"),
    extensions: {vector},
});

await pglite.exec("CREATE EXTENSION IF NOT EXISTS vector;");
await pglite.exec(`
  CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536)
  );
  CREATE INDEX ON articles USING hnsw (embedding vector_cosine_ops);
`);

const rows = await pglite.query("select * from articles;");

console.log(`rows: ${JSON.stringify(rows)}`)

// 
// console.log(JSON.stringify(rows))

// const pg = new PGlite('opfs-ahp://Users/shuntaka/repos/github.com/shuntaka9576/plamo-embed/ts/db')