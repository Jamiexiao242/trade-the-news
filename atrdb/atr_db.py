import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, List

class ATRDatabase:
    """本地ATR数据缓存数据库"""
    
    def __init__(self, db_path: str = "atr_cache.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            cols = conn.execute("PRAGMA table_info(atr_data)").fetchall()
            col_names = {c[1] for c in cols}
            if col_names and col_names != {"ticker", "atr"}:
                # Migrate legacy schema to the simplified ticker/atr table.
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS atr_data_new (
                        ticker TEXT PRIMARY KEY,
                        atr REAL NOT NULL
                    )
                """)
                conn.execute("""
                    INSERT OR REPLACE INTO atr_data_new (ticker, atr)
                    SELECT ticker, atr FROM atr_data
                """)
                conn.execute("DROP TABLE atr_data")
                conn.execute("ALTER TABLE atr_data_new RENAME TO atr_data")
                conn.commit()
                return
            conn.execute("""
                CREATE TABLE IF NOT EXISTS atr_data (
                    ticker TEXT PRIMARY KEY,
                    atr REAL NOT NULL
                )
            """)
            conn.commit()
    
    def upsert(self, ticker: str, atr: float) -> None:
        """插入或更新ATR数据"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO atr_data 
                (ticker, atr)
                VALUES (?, ?)
            """, (ticker, atr))
            conn.commit()
    
    def get(self, ticker: str) -> Optional[Dict]:
        """获取ATR数据"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM atr_data WHERE ticker = ?
            """, (ticker,))
            
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row)
    
    def get_all(self) -> List[Dict]:
        """获取所有ATR数据"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM atr_data")
            return [dict(row) for row in cursor.fetchall()]
    
    def stats(self) -> Dict:
        """获取数据库统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    AVG(atr) as avg_atr
                FROM atr_data
            """)
            row = cursor.fetchone()
            
        return {
            'total_tickers': row[0],
            'avg_atr': round(row[1], 6) if row[1] else 0.0
        }


# JSON版本（更简单，适合小规模）
class ATRDatabaseJSON:
    """JSON文件版本（更简单）"""
    
    def __init__(self, json_path: str = "atr_cache.json"):
        self.json_path = Path(json_path)
        self._load()
    
    def _load(self):
        if self.json_path.exists():
            with open(self.json_path) as f:
                self.data = json.load(f)
        else:
            self.data = {}
    
    def _save(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def upsert(self, ticker: str, atr: float) -> None:
        self.data[ticker] = {
            'atr': atr
        }
        self._save()
    
    def get(self, ticker: str) -> Optional[Dict]:
        if ticker not in self.data:
            return None
        return self.data[ticker]
    
    def get_all(self) -> Dict:
        return self.data
    
    def stats(self) -> Dict:
        if not self.data:
            return {'total_tickers': 0}

        return {
            'total_tickers': len(self.data)
        }
