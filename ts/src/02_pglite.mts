import { PGlite } from '@electric-sql/pglite';

// const db = new PGlite();
// const rows = await db.query("select 'hello world' as message;")
// 
// console.log(JSON.stringify(rows))

const pg = new PGlite('opfs-ahp://Users/shuntaka/repos/github.com/shuntaka9576/plamo-embed/ts/db')