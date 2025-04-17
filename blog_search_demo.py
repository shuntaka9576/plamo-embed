import sqlite3
import sqlite_vss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Any
from datetime import datetime

# PLaMo埋め込みモデルのロード
tokenizer = AutoTokenizer.from_pretrained(
    "pfnet/plamo-embedding-1b", trust_remote_code=True
)
model = AutoModel.from_pretrained("pfnet/plamo-embedding-1b", trust_remote_code=True)
device = "cpu"
model = model.to(device)


def connect_db(db_path="blog_search.db"):
    """データベースに接続する"""
    conn = sqlite3.connect(db_path, timeout=10)
    conn.enable_load_extension(True)
    sqlite_vss.load(conn)
    vss_version = conn.execute("SELECT vss_version()").fetchone()[0]
    print(f"SQLite VSS バージョン: {vss_version}")
    return conn


def setup_database(conn):
    """データベーステーブルの初期設定"""
    cursor = conn.cursor()

    # ブログ記事テーブルの作成
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS blog_posts(
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        url TEXT NOT NULL,
        created_at DATETIME
    );
    """)

    # ベクトル検索テーブルの作成 - PLaMoの埋め込みは2048次元
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS vss_blog USING vss0(
        content_embedding(2048)
    );
    """)

    # サンプルデータの挿入
    sample_data = [
        # ITカテゴリ
        {
            "title": "機械学習入門：基本概念とアルゴリズム",
            "content": "機械学習は人工知能の一分野で、コンピュータがデータから学習し予測や判断を行う技術です。教師あり学習、教師なし学習、強化学習などの種類があります。",
            "url": "https://example.com/ml-intro",
            "created_at": datetime.now(),
        },
        {
            "title": "深層学習の基礎：ニューラルネットワーク入門",
            "content": "深層学習はニューラルネットワークを多層化した機械学習手法です。画像認識、自然言語処理、音声認識など様々な分野で革命的な進歩をもたらしています。",
            "url": "https://example.com/deep-learning",
            "created_at": datetime.now(),
        },
        {
            "title": "自然言語処理の最新トレンド",
            "content": "自然言語処理は人間の言語をコンピュータに理解させる技術です。Transformer、BERTなどのモデルの登場により、機械翻訳や質問応答システムが大きく進化しました。",
            "url": "https://example.com/nlp-trends",
            "created_at": datetime.now(),
        },
        # 医療カテゴリ
        {
            "title": "最新のがん治療法とその効果",
            "content": "がん治療は免疫療法や標的治療など革新的な手法が開発されています。従来の化学療法や放射線治療と比較して、より特異的で副作用の少ない治療が可能になってきました。",
            "url": "https://example.com/cancer-treatment",
            "created_at": datetime.now(),
        },
        {
            "title": "生活習慣病の予防と管理",
            "content": "糖尿病、高血圧、脂質異常症などの生活習慣病は適切な食事管理と運動で予防・改善できます。定期的な健康診断により早期発見と適切な介入が重要です。",
            "url": "https://example.com/lifestyle-disease",
            "created_at": datetime.now(),
        },
        {
            "title": "医療AIの進化と倫理的課題",
            "content": "医療分野でのAI活用は画像診断支援や治療計画の最適化など様々な場面で進んでいます。しかし、診断の責任や患者データのプライバシーなど、倫理的課題も多く存在します。",
            "url": "https://example.com/medical-ai",
            "created_at": datetime.now(),
        },
        # 金融カテゴリ
        {
            "title": "投資初心者のための資産運用入門",
            "content": "資産運用は分散投資とリスク管理が基本です。株式、債券、不動産など異なる資産クラスへ配分することで、市場変動の影響を抑えつつ安定したリターンを目指せます。",
            "url": "https://example.com/investment-basics",
            "created_at": datetime.now(),
        },
        {
            "title": "暗号資産の仕組みと将来性",
            "content": "ビットコインやイーサリアムなどの暗号資産はブロックチェーン技術を基盤としています。分散型金融（DeFi）やNFTなど、新たな応用領域も広がっています。",
            "url": "https://example.com/crypto-future",
            "created_at": datetime.now(),
        },
        {
            "title": "フィンテックがもたらす金融革命",
            "content": "フィンテックは金融とテクノロジーを融合した新サービスです。モバイル決済、ロボアドバイザー、クラウドファンディングなど、従来の銀行システムを変革しています。",
            "url": "https://example.com/fintech-revolution",
            "created_at": datetime.now(),
        },
    ]

    for post in sample_data:
        cursor.execute(
            """
        INSERT INTO blog_posts(title, content, url, created_at)
        VALUES (?, ?, ?, ?)
        """,
            (post["title"], post["content"], post["url"], post["created_at"]),
        )

        # 挿入されたレコードのIDを取得
        last_id = cursor.lastrowid

        # コンテンツのベクトル埋め込みを生成
        content_embedding = generate_embedding(post["content"])

        # vss_blogテーブルに埋め込みベクトルを挿入
        cursor.execute(
            """
        INSERT INTO vss_blog(rowid, content_embedding)
        VALUES (?, ?)
        """,
            (last_id, serialize_vector(content_embedding)),
        )

    conn.commit()
    print("データベースのセットアップが完了しました。")


def generate_embedding(text: str) -> np.ndarray:
    """PLaMoモデルを使用してテキストの埋め込みベクトルを生成する"""
    with torch.inference_mode():
        embedding = model.encode_document([text], tokenizer)
        # numpyに変換して返す
        return embedding.cpu().numpy().squeeze()


def serialize_vector(vector: np.ndarray) -> bytes:
    """ベクトルをバイト形式にシリアライズする"""
    return np.asarray(vector).astype(np.float32).tobytes()


def search(conn, query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """セマンティック検索を実行し、関連性の高いブログ記事を返す"""
    # クエリは検索用の埋め込みを使用
    with torch.inference_mode():
        query_embedding = model.encode_query(query, tokenizer)
        query_embedding = query_embedding.cpu().numpy().squeeze()

    cursor = conn.cursor()
    cursor.execute(
        """
    SELECT 
        blog_posts.id,
        blog_posts.title,
        blog_posts.content,
        blog_posts.url,
        blog_posts.created_at,
        vss_blog.distance
    FROM vss_blog
    JOIN blog_posts ON vss_blog.rowid = blog_posts.id
    WHERE vss_search(content_embedding, vss_search_params(?, 10))
    ORDER BY distance ASC
    LIMIT ?
    """,
        (serialize_vector(query_embedding), limit),
    )

    results = []
    for row in cursor.fetchall():
        id, title, content, url, created_at, distance = row
        similarity = 1.0 - float(distance)
        results.append(
            {
                "id": id,
                "title": title,
                "content": content[:100] + "..." if len(content) > 100 else content,
                "url": url,
                "created_at": created_at,
                "similarity": similarity,
            }
        )

    return results


def main():
    # データベース接続
    conn = connect_db()

    # データベースセットアップ
    setup_database(conn)

    # 検索クエリの例
    example_queries = [
        # ITカテゴリの検索クエリ
        "機械学習について教えて",
        "ニューラルネットワークの仕組み",
        # 医療カテゴリの検索クエリ
        "最新のがん治療について知りたい",
        "生活習慣病の予防法",
        # 金融カテゴリの検索クエリ
        "初心者向け投資方法",
        "暗号通貨とブロックチェーン",
    ]

    print("\n===== セマンティック検索デモ =====")
    for query in example_queries:
        print(f"\n検索クエリ: '{query}'")
        results = search(conn, query)

        print(f"検索結果 (上位 {len(results)} 件):")
        for i, result in enumerate(results):
            print(f"{i + 1}. {result['title']} - 類似度: {result['similarity']:.4f}")
            print(f"   URL: {result['url']}")
            print(f"   概要: {result['content']}")

    # データベース接続を閉じる
    conn.close()


if __name__ == "__main__":
    main()
