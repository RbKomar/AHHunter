import sqlite3
import logging
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path='ah_prices.db'):
        self.db_path = db_path
        self.conn = None
        try:
            self._connect()
            self._create_tables()
        except sqlite3.Error as e:
            logging.critical(f"Database initialization failed: {e}")
            raise

    def _connect(self):
        self.conn = sqlite3.connect(self.db_path)
        logging.info(f"Successfully connected to database at {self.db_path}")

    def _create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id INTEGER NOT NULL,
                    price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (item_id) REFERENCES items (id)
                )
            """)
            logging.info("Database tables created or already exist.")

    def get_item_id(self, item_name, create_if_not_exists=True):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM items WHERE name = ?", (item_name,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        elif create_if_not_exists:
            cursor.execute("INSERT INTO items (name) VALUES (?)", (item_name,))
            self.conn.commit()
            logging.info(f"Created new item in database: {item_name}")
            return cursor.lastrowid
        else:
            return None

    def get_latest_price(self, item_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT price FROM price_history
            WHERE item_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (item_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def update_price_if_changed(self, item_name, new_price):
        item_id = self.get_item_id(item_name)
        if item_id is None:
            return False 

        latest_price = self.get_latest_price(item_id)

        if latest_price is None or not abs(latest_price - new_price) < 1e-9:
            with self.conn:
                self.conn.execute("""
                    INSERT INTO price_history (item_id, price, timestamp)
                    VALUES (?, ?, ?)
                """, (item_id, new_price, datetime.utcnow().isoformat()))
            logging.debug(f"Updated price for {item_name} to {new_price}")
            return True
        return False

    def get_price_history(self, item_name):
        item_id = self.get_item_id(item_name, create_if_not_exists=False)
        if not item_id:
            return []
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT timestamp, price FROM price_history
            WHERE item_id = ?
            ORDER BY timestamp ASC
        """, (item_id,))
        return cursor.fetchall()


    def close(self):
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")
